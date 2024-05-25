# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
import copy
from contextlib import contextmanager
from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ProcessGroup
from torch.optim import Optimizer
from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import (
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_DATA_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
)

from .utils import compute_norm


from internlm.solver.optimizer.store import (
    BucketStore_v2,
    GradientStore_v2,
    ParameterStore_v2,
)

from internlm.solver.optimizer.utils import (
    DynamicGradScaler,
    flatten,
    has_inf_or_nan,
    reduce_tensor,
    release_param_grad,
    sync_param,
    
)

from internlm.utils.parallel import is_using_isp, is_using_sequence_parallel


from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger
from internlm.monitor import send_alert_message

import math


def calculate_global_norm_from_list(norm_list):
    """Compute total from a list of norms"""
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm**2.0
    return math.sqrt(total_norm)


from torch.optim import Optimizer
from internlm.core.parallel.comm.zero import ParamAsyncBcastHandler

from internlm.core.context import Config, ParallelMode
from .base_optimizer import BaseOptimizer

logger = get_logger(__file__)


class HybridZeroOptimizer_v2(BaseOptimizer):
    """Optimizer used for ZeRO-1 and ZeRO-2."""

    def __init__(
        self,
        optimizer: Optimizer,
        grad_scal_cfg: Config = None,
        zero_cfg: Config = None,
        param_bcast_sync_handler: ParamAsyncBcastHandler = None,
        isp_communicator=None,
        communication_dtype: Optional[torch.dtype] = None,
        partition_grad: bool = False,  # stage 2 flag
        cpu_offload: bool = False,  # cpu offload
        forced_dtype: Optional[torch.dtype] = None,
        moe_extra_dp_process_group: Optional[ProcessGroup] = None,
        master_weights: bool = True,  # master weights
        
    ):
        print(f"optimizer_version: HybridZeroOptimizer_v2", flush=True)
        if gpc.config.model.dtype is torch.float32:
            initial_scale = 1
        else:
            initial_scale = grad_scal_cfg.fp16.initial_scale
        min_scale = grad_scal_cfg.fp16.min_scale
        growth_interval = grad_scal_cfg.fp16.growth_interval
        growth_factor = grad_scal_cfg.growth_factor
        backoff_factor = grad_scal_cfg.backoff_factor
        hysteresis = grad_scal_cfg.hysteresis
        max_scale = grad_scal_cfg.max_scale

        # Zero related args
        reduce_bucket_size = zero_cfg.reduce_bucket_size
        clip_grad_norm = zero_cfg.clip_grad_norm
        self._overlap_sync_grad = zero_cfg.overlap_sync_grad
        self._overlap_sync_param = zero_cfg.overlap_sync_param
        self.use_isp = is_using_isp()
        
        self._param_bcast_sync_handler = param_bcast_sync_handler
        
        if self._overlap_sync_param:
            assert self._param_bcast_sync_handler is not None

        self._isp_communicator = isp_communicator
        
        super().__init__(optim=optimizer)
        

        self._dtype = self.optim.param_groups[0]["params"][0].dtype

        # stage 2
        self._partition_grads = partition_grad

        self._cpu_offload = cpu_offload

        # grad accumulation
        self.require_grad_sync = True

        # if process_group is none, will use the default one
        self._local_rank = gpc.get_local_rank(ParallelMode.DATA)
        self._world_size = gpc.get_world_size(ParallelMode.DATA)

        self._zero_local_rank = []
        self._zero_world_size = []
        self._broadcast_parallel_mode = []
        self._group_id_map_zero_mode = dict()
        
        # extra dp
        # This group is used to sync moe param, dp_world_size = moe_duplicates * extra_dp_size.
        # Non moe param will be sync by global dp pg, moe param will be sync by extra dp pg.
        # Moe param grad is be split as non moe param by global dp pg, and grad will be merged in step.
        # And moe working and master param are split by extra dp pg.
        self.moe_extra_dp_pg = moe_extra_dp_process_group
        if self.moe_extra_dp_pg is not None:
            self.moe_extra_dp_pg_size = dist.get_world_size(group=self.moe_extra_dp_pg)
            self.moe_extra_dp_pg_rank = dist.get_rank(group=self.moe_extra_dp_pg)

        # working and master params for mixed precision training
        self._working_param_groups = dict()
        self._master_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = zero_cfg.overlap_sync_grad
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype
        
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
        )
        
        self.found_overflow = torch.zeros(1, dtype=torch.float, device=get_current_device())

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm
        
        # master weights copy
        self._master_weights = master_weights

        if forced_dtype:
            for group in self.optim.param_groups:
                group_params = group["params"]
                for param in group_params:
                    param.data = param.data.to(forced_dtype)
            self._dtype = forced_dtype

        # check argument conflict
        self._sanity_checks()

        # ParameterStore_v2 will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        parallel_mode = ParallelMode.WEIGHT_DATA if self.use_isp else ParallelMode.DATA
        self._param_store = ParameterStore_v2(ParallelMode.ZERO1)
        self._grad_store = GradientStore_v2(parallel_mode, partition_grad=partition_grad)
        self._bucket_store: List[BucketStore_v2] = []
        self._accum_grad_buckets: List[BucketStore_v2] = []

        # moe param should not be stored in working_groups
        # because they have different parallel strategy
        # so we need to store them separately in param_groups
        # instead of working_groups
        self.working_moe_params = list()
        
        self.rank_unique_id = (
            f"gpus-{gpc.get_world_size(ParallelMode.GLOBAL)}_"
            + f"wp-{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
            + f"tp-{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"dp-{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"pp-{gpc.get_local_rank(ParallelMode.PIPELINE)}_"
            + f"zo-{gpc.get_local_rank(ParallelMode.ZERO1)}.pt"
        )

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = list()
            for param in param_group["params"]:
                if param.requires_grad:
                    setattr(param, "group_id", group_id)
                    group_params.append(param)
                    
            param_group["dtype"] = group_params[0].dtype if len(group_params) != 0 else None

            zero_mode = param_group["optimizer_mode"]
            self._zero_local_rank.append(gpc.get_local_rank(zero_mode))
            self._zero_world_size.append(gpc.get_world_size(zero_mode))
            self._broadcast_parallel_mode.append(zero_mode)
            self._group_id_map_zero_mode[group_id] = zero_mode

            # add the working params to working_param_groups for bookkeeping
            self._working_param_groups[group_id] = group_params
            master_param_current_rank = self._create_master_param_current_rank(group_id, group_params)
            self._master_param_groups_of_current_rank[group_id] = master_param_current_rank

            # need to replace the params in the `params` field in the optimizer
            # so that when the optimizer calls step(), it only updates the tensors
            # managed by this data parallel rank
            param_group["params"] = master_param_current_rank
            
            if self._is_moe_group(param_group):
                grad_reduce_mode = ParallelMode.EXPERT_DATA
            elif param_group["name"] != "embed_head" and self.use_isp:
                grad_reduce_mode = ParallelMode.WEIGHT_DATA
            else:
                grad_reduce_mode = ParallelMode.DATA
            self._bucket_store.append(BucketStore_v2(group_id, grad_reduce_mode, zero_mode=zero_mode))
            self._accum_grad_buckets.append(BucketStore_v2(group_id, grad_reduce_mode, zero_mode=zero_mode))

        # if there are moe params, store in addtional group in optim
        if len(self.working_moe_params) > 0:
            self._sync_master_param = False
            param_group = dict()
            # create fp32 master param
            for key, value in self.optim.param_groups[0].items():
                if key != "params":
                    param_group[key] = value
            self.master_moe_params = []
            for param in self.working_moe_params:
                self.master_moe_params.append(param.clone().to(torch.float32).detach())
            # create mapping from master to working for optimizer io
            self.moe_master_to_working_map = {}
            for master_moe_param, working_moe_param in zip(self.master_moe_params, self.working_moe_params):
                self.moe_master_to_working_map[id(master_moe_param)] = working_moe_param
            # add to optim
            param_group["params"] = self.master_moe_params
            self.optim.param_groups.append(param_group)

        # initialize communication stream for
        # communication-computation overlapping
        if self._overlap_communication:
            self._comm_stream = torch.cuda.Stream(priority=0)
            
        ### todo    
        self.skip_grad_reduce = False

        # InternEvo origin hook    
        self._attach_reduction_hook()        
    

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_param_groups(self):
        return len(self._working_param_groups)
    
    @property
    def loss_scale(self) -> float:
        return self.grad_scaler.scale.item()
    
    def _is_moe_group(self, param_group):
        return "moe" in param_group.keys() and param_group["moe"]
    
    def _wait_reduce_scatter_and_accumulate_grads(self, param):
        param_size = param.numel()

        group_id = getattr(param, "group_id")
        current_bucket = self._accum_grad_buckets[group_id]

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # after reduction, the bucket will be empty
        if current_bucket.num_elements_in_bucket() + param_size > self._reduce_bucket_size:
            self._accum_grads_store_in_bucket(current_bucket)
            
        # otherwise, add the parameter into bucket.
        current_bucket._num_elements_in_bucket += param.numel()
        current_bucket._param_list.append(param)
        
    def _accum_grads_store_in_bucket(self, bucket: BucketStore_v2) -> None:
        for _param in bucket.get_param():
            if not hasattr(_param, "isp_reduce_scatter_name"):
                continue

            # wait and accumulate gardient.
            _key = getattr(_param, "isp_reduce_scatter_name")
            _grad, _comm_handle = self._isp_communicator.reduce_scatter_handlers[_key]
            _comm_handle.wait()
            _param.grad.add_(_grad)

            # release cuda memory.
            if self._isp_communicator.enable_memory_pool:
                self._isp_communicator.memory_pool.free_reduce_scatter_memory(
                    key=tuple(_grad.size()), index=_grad.index
                )
            _grad = None
            self._isp_communicator.reduce_scatter_handlers[_key] = None

        bucket.reset_all()
        
    def accumulate_left_grads_after_backward(self):
        if self._isp_communicator is None or self._isp_communicator.overlap is False:
            return
        
        for group_id in range(self.num_param_groups):
            self._accum_grads_store_in_bucket(self._accum_grad_buckets[group_id])
    
    def should_skip_step(self) -> bool:
        found_inf = self.check_overflow()
        self.grad_scaler.update(found_inf)
        return found_inf
    
    def check_overflow(self) -> bool:
        # clear previous overflow record
        self.found_overflow.fill_(0.0)
        if self.check_local_overflow():
            self.found_overflow.fill_(1.0)
        dist.all_reduce(self.found_overflow, op=dist.ReduceOp.MAX)
        return self.found_overflow.item() > 0
    
    def check_local_overflow(self) -> bool:
        for group_id in range(self.num_param_groups):
            for avg_grad in self._grad_store.get_working_grads_by_group_id(group_id, self._group_id_map_zero_mode[group_id]):
                if avg_grad is not None and has_inf_or_nan(avg_grad):
                    return True
        return False
    
    def clip_grad_norm(self, model, max_norm):
        # will conduct in the step()
        pass

    def _sanity_checks(self):
        # assert get_accelerator().name in ["cuda", "npu"], "device is required"
        for param_group in self.optim.param_groups:
            group_params = param_group["params"]
            for param in group_params:
                if not hasattr(param, "skip_zero_check") or param.skip_zero_check is False:
                    assert (
                        param.dtype == self._dtype
                    ), f"Parameters are expected to have the same dtype `{self._dtype}`, but got `{param.dtype}`"
    
                    
    def add_hook_for_splited_param(self, origin_param, splited_param_current_rank):
        
        if hasattr(origin_param, IS_TENSOR_ZERO_PARALLEL):
            value = getattr(origin_param, IS_TENSOR_ZERO_PARALLEL)
            setattr(splited_param_current_rank, IS_TENSOR_ZERO_PARALLEL, value)
            
        if hasattr(origin_param, IS_WEIGHT_ZERO_PARALLEL):
            value = getattr(origin_param, IS_WEIGHT_ZERO_PARALLEL)
            setattr(splited_param_current_rank, IS_WEIGHT_ZERO_PARALLEL, value)
            
        if hasattr(origin_param, IS_REPLICA_ZERO_PARALLEL):
            value = getattr(origin_param, IS_REPLICA_ZERO_PARALLEL)
            setattr(splited_param_current_rank, IS_REPLICA_ZERO_PARALLEL, value)
        
        if hasattr(origin_param, IS_TENSOR_DATA_PARALLEL):
            value = getattr(origin_param, IS_TENSOR_DATA_PARALLEL)
            setattr(splited_param_current_rank, IS_TENSOR_DATA_PARALLEL, value)
        
        if hasattr(origin_param, IS_TENSOR_EXPERT_DATA_PARALLEL):
            value = getattr(origin_param, IS_TENSOR_EXPERT_DATA_PARALLEL)
            setattr(splited_param_current_rank, IS_TENSOR_EXPERT_DATA_PARALLEL, value)
                

    def _create_master_param_current_rank(self, group_id, param_list):
        # split each param evenly by world size
        params_current_rank = []
        device = "cpu" if self._cpu_offload else get_current_device()
        zero_world_size = self._zero_world_size[group_id]
        
        for param in param_list:
            padding_size = (zero_world_size - param.numel() % zero_world_size) % zero_world_size
            self._param_store.record_param_padding_size(param, padding_size)

            with torch.no_grad():
                if padding_size > 0:
                    padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
                    # reset working params' ptr when no master weights
                    if self._master_weights == False:
                        param.data = padding_param[: param.numel()].view(param.shape)
                else:
                    padding_param = param.data.view(-1)

                splited_params = padding_param.split(padding_param.numel() // zero_world_size)
                splited_params = splited_params[self._zero_local_rank[group_id]]

                # use fp32 when master_weights is True
                if self._master_weights is True:
                    splited_param_current_rank = splited_params.detach().float().to(device)
                else:
                    splited_param_current_rank = splited_params
                    
                self.add_hook_for_splited_param(param, splited_param_current_rank)
                
                params_current_rank.append(splited_param_current_rank)
                self._param_store.link_master_and_working_param(splited_param_current_rank, param)
                

        return params_current_rank

    #######################
    # Reduction Functions #
    #######################

    def _run_reduction(self):
        for group_id in range(self.num_param_groups):
            current_bucket = self._bucket_store[group_id]
            dp_parallel_mode = current_bucket.get_dp_parallel_mode()
            reduce_group = gpc.get_group(dp_parallel_mode)
            world_size = gpc.get_world_size(dp_parallel_mode)
            if current_bucket.num_elements_in_bucket() > 0:
                current_bucket.build_grad_in_bucket()

                if self.moe_extra_dp_pg is None:
                    flat_grads = current_bucket.get_flatten_grad()
                    flat_grads /= world_size
                else:
                    # record moe and non moe param
                    moe_list = []


                    # divide them into different groups
                    moe_grad_list = []
                    non_moe_grad_list = []
                    for grad_list in current_bucket._grad_in_bucket.values():
                        non_moe_cur_grad = []
                        moe_cur_grad = []
                        for i in range(len(grad_list)):
                            if moe_list[i] == True:
                                moe_cur_grad.append(grad_list[i])
                            else:
                                non_moe_cur_grad.append(grad_list[i])
                        if len(moe_cur_grad) > 0:
                            moe_grad_list.append(moe_cur_grad)
                        if len(non_moe_cur_grad) > 0:
                            non_moe_grad_list.append(non_moe_cur_grad)

                    if len(non_moe_grad_list) > 0:
                        non_moe_flat_grads = []
                        for grad_list in non_moe_grad_list:
                            non_moe_flat_grads.append(_flatten_dense_tensors(grad_list))
                        non_moe_flat_grads = _flatten_dense_tensors(non_moe_flat_grads)
                        non_moe_flat_grads /= world_size

                    if len(moe_grad_list) > 0:
                        moe_flat_grads = []
                        for grad_list in moe_grad_list:
                            moe_flat_grads.append(_flatten_dense_tensors(grad_list))
                        moe_flat_grads = _flatten_dense_tensors(moe_flat_grads)

                # ready to add other tensors to bucket
                current_bucket.reset_num_elements_in_bucket()

                if self._overlap_communication:
                    stream = self._comm_stream
                    # in case of the memory being reused in the default stream
                    if self.moe_extra_dp_pg is None:
                        flat_grads.record_stream(stream)
                    else:
                        if len(non_moe_grad_list) > 0:
                            non_moe_flat_grads.record_stream(stream)
                        if len(moe_grad_list) > 0:
                            moe_flat_grads.record_stream(stream)
                    # waiting for ops in the default stream finishing
                    stream.wait_stream(torch.cuda.current_stream())
                else:
                    stream = torch.cuda.current_stream()

                with torch.cuda.stream(stream):
                    group_id = current_bucket.get_param_group_id()

                    if self.moe_extra_dp_pg is None:
                        grad_dtype = flat_grads.dtype
                        if self._communication_dtype is not None:
                            flat_grads = flat_grads.to(self._communication_dtype)

                    if not self._partition_grads:
                        if self.moe_extra_dp_pg is None:
                            dist.all_reduce(flat_grads, group=reduce_group)
                            if flat_grads.dtype != grad_dtype:
                                flat_grads = flat_grads.to(grad_dtype)

                            flat_grads_per_rank = flat_grads.split(flat_grads.numel() // self._zero_world_size[group_id])
                            grad_in_bucket = current_bucket.get_grad()
                            self._update_unpartitoned_grad(grad_in_bucket.values(), flat_grads_per_rank, group_id)

                        # sync extra zero group
                        else:
                            # sync non moe param in global dp group
                            if len(non_moe_grad_list) > 0:
                                dist.all_reduce(non_moe_flat_grads, group=reduce_group)
                                flat_grads_per_rank = non_moe_flat_grads.split(
                                    non_moe_flat_grads.numel() // self._zero_world_size[group_id]
                                )
                                self._update_unpartitoned_grad(non_moe_grad_list, flat_grads_per_rank, group_id)

                            # sync moe param only in zero group
                            if len(moe_grad_list) > 0:
                                dist.all_reduce(moe_flat_grads, group=self.moe_extra_dp_pg)
                                flat_grads_per_rank = moe_flat_grads.split(moe_flat_grads.numel() // self._zero_world_size[group_id])
                                self._update_unpartitoned_grad(moe_grad_list, flat_grads_per_rank, group_id)

                    else:
                        if self.moe_extra_dp_pg is None:
                            flat_grads_list = list(flat_grads.split(len(flat_grads) // world_size))
                            recieved_grad = torch.zeros_like(flat_grads_list[0])
                            dist.reduce_scatter(recieved_grad, flat_grads_list, group=reduce_group)

                            if recieved_grad.dtype != grad_dtype:
                                recieved_grad = recieved_grad.to(grad_dtype)

                            grad_in_bucket_current_rank = current_bucket.get_grad()[self._zero_local_rank[group_id]]
                            self._update_partitoned_grad(grad_in_bucket_current_rank, recieved_grad, group_id, 1)
                        else:
                            # categorize moe and non moe param
                            grad_in_bucket_current_rank = current_bucket.get_grad()[self._zero_local_rank[group_id]]
                            moe_grad_in_bucket_current_rank = []
                            non_moe_grad_in_bucket_current_rank = []
                            for idx, grad in enumerate(grad_in_bucket_current_rank):
                                if moe_list[idx] == True:
                                    moe_grad_in_bucket_current_rank.append(grad)
                                else:
                                    non_moe_grad_in_bucket_current_rank.append(grad)

                            if len(non_moe_grad_list) > 0:
                                flat_grads_list = list(
                                    non_moe_flat_grads.split(len(non_moe_flat_grads) // world_size)
                                )
                                recieved_grad = torch.zeros_like(flat_grads_list[0])
                                dist.reduce_scatter(recieved_grad, flat_grads_list, group=reduce_group)
                                self._update_partitoned_grad(
                                    non_moe_grad_in_bucket_current_rank,
                                    recieved_grad,
                                    group_id,
                                    1,
                                )

                            if len(moe_grad_list) > 0:
                                flat_grads_list = list(
                                    moe_flat_grads.split(len(moe_flat_grads) // self.moe_extra_dp_pg_size)
                                )
                                recieved_grad = torch.zeros_like(flat_grads_list[0])
                                dist.reduce_scatter(
                                    recieved_grad,
                                    flat_grads_list,
                                    group=self.moe_extra_dp_pg,
                                )
                                param_slice = self._world_size // self.moe_extra_dp_pg_size
                                recieved_grad = list(recieved_grad.split(len(recieved_grad) // param_slice))
                                for split_recieved_grad in recieved_grad:
                                    split_recieved_grad = _unflatten_dense_tensors(
                                        split_recieved_grad, moe_grad_in_bucket_current_rank
                                    )
                                    for real_grad, grad in zip(split_recieved_grad, moe_grad_in_bucket_current_rank):
                                        param_id = current_bucket.get_param_id_of_grad(grad)
                                        self._add_grad(real_grad, param_slice, group_id, param_id)

                    current_bucket.reset()

    def _update_unpartitoned_grad(self, origin_grad_list: List, flat_grad_list: List, group_id: int) -> None:
        for rank, grad_list in enumerate(origin_grad_list):
            sync_param(flat_grad_list[rank], grad_list)
            for grad in grad_list:
                if grad == None:
                    print("_update_unpartitoned_grad grad None", flush=True)
                param_id = self._bucket_store[group_id].get_param_id_of_grad(grad)
                self._add_grad(grad, self._zero_world_size[group_id], group_id, param_id, rank)

    def _update_partitoned_grad(
        self,
        origin_grad_list: List,
        flat_grad: torch.Tensor,
        group_id: int,
        partition_num: int,
    ) -> None:
        sync_param(flat_grad, origin_grad_list)
        for grad in origin_grad_list:
            param_id = self._bucket_store[group_id].get_param_id_of_grad(grad)
            self._add_grad(grad, partition_num, group_id, param_id)

    def _add_grad(
        self,
        grad: torch.Tensor,
        partition_num: int,
        group_id: int,
        param_id: int,
        rank: int = 0,
    ) -> None:
        if len(self._grad_store.get_partitioned_gradients_by_param_id(group_id, param_id, "_add_grad")) < partition_num:
            self._grad_store.append_gradients_by_param_id(grad, group_id, param_id)
        else:
            self._grad_store.add_gradients_by_param_id(grad, rank, group_id, param_id)

    def _add_to_bucket(self, param, group_id):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # or got a grad of param from another group
        # after reduction, the bucket will be empty
        if (
            self._bucket_store[group_id].num_elements_in_bucket() + param_size > self._reduce_bucket_size
            or group_id != self._bucket_store[group_id].get_param_group_id()
        ):
            self._run_reduction()

        padding_size = self._param_store.get_param_padding_size(param)
        self._bucket_store[group_id].add_param_grad(param, padding_size)

    ################################
    # torch.optim.Optimizer methods
    ################################

    def backward(self, loss, retain_graph=False):
        assert not (
            self._partition_grads and not self.require_grad_sync
        ), "ZeRO2(partition_grads) and no_sync are not compatible"     
        
        loss = self.loss_scale * loss

        loss.backward(retain_graph=retain_graph)
        

        if not self.require_grad_sync:
            return
        
        self.accumulate_left_grads_after_backward()
        self._reduce_grad(self._partition_grads)
        

        # clear reduced grads
        if self._overlap_communication:
            torch.cuda.synchronize()
        self.zero_grad()

    def backward_by_grad(self, tensor, grad):
        assert not (
            self._partition_grads and not self.require_grad_sync
        ), "ZeRO2(partition_grads) and gradient accumulation(no_sync) are not compatible"

        torch.autograd.backward(tensor, grad)

        if not self.require_grad_sync:
            return
        
        self.accumulate_left_grads_after_backward()
        self._reduce_grad(self._partition_grads)

        # clear reduced grads
        if self._overlap_communication:
            torch.cuda.synchronize()

        self.zero_grad()

    def zero_grad(self, set_to_none=True):
        """
        Set parameter gradients to zero. If set_to_none = True, gradient
        will be set to None to save memory.

        :param set_to_none: Whether set the gradient to None. Default value is True.
        :type set_to_none: bool
        """
        for _, param_group in self._working_param_groups.items():
            for param in param_group:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach()
                        param.grad.zero_()
        for group_id in range(self.num_param_groups):                
            self._bucket_store[group_id].reset_all()

    ####################
    # Update Parameter #
    ####################

    def step(self, closure=None):
        assert closure is None, "closure is not supported by step()"
        if not self.require_grad_sync:
            return

        if self.should_skip_step():
            self._grad_store.reset_all_gradients()
            print("Found overflow. Skip step", flush=True)
            self.zero_grad()
            return

        # record all grads for unscale and clip
        grad_partition_groups = []
        norm_groups = []

        # sometimes not all params are 'really' working
        # for instance, when layer drop, the dropped layer has no grad
        # and should not be updated
        real_working_params = dict()
        real_master_params = dict()
        real_master_grads = dict()
        
        total_norms = {}
        single_grad_partition_groups = []

        
        for group_id in range(self.num_param_groups):
            master_params = self._master_param_groups_of_current_rank[group_id]
            real_working_params[group_id] = []
            real_master_params[group_id] = []
            real_master_grads[group_id] = []
            fp32_avg_grads = []
            grad_index = 0 if self._partition_grads else self._zero_local_rank[group_id]
            
            
            for splited_param in master_params:
                working_param = self._param_store.master_to_working_param[id(splited_param)]
                # if a working param requires grad and has no grad
                # it is not 'really' working, e.g. the droped layer
                # else the splited grad should be attached to the splited param
                grads = self._grad_store.get_partitioned_gradients_by_param_id(group_id, id(working_param), "working_param")
                if len(grads) == 0:
                    print(f"grads: {gpc.get_local_rank(ParallelMode.PIPELINE)}, len(grads): {len(grads)}, group_id: {group_id}", flush=True)
                if len(grads) > 0:
                    # moe hybrid zero
                    if self.moe_extra_dp_pg is not None and False:
                        real_working_params[group_id].append(working_param)
                        if self._partition_grads:
                            grad = grads
                        else:
                            param_slice = self._zero_world_size[group_id] // self.moe_extra_dp_pg_size
                            grad = grads[
                                self.moe_extra_dp_pg_rank * param_slice : (self.moe_extra_dp_pg_rank + 1) * param_slice
                            ]
                        grad = flatten(grad)
                    else:
                        real_working_params[group_id].append(working_param)
                        grad = grads[grad_index]
                    # no need to copy fp32 grad if master_weights is False
                    if self._master_weights:
                        grad = grad.to(splited_param.dtype).to(splited_param.device)
                    splited_param.grad = grad
                    grad_partition_groups.append(grad)
                    fp32_avg_grads.append(grad)
                    real_master_params[group_id].append(splited_param)
                    real_master_grads[group_id].append(splited_param.grad)
                 
            
            if fp32_avg_grads != []:
                # single_grad_partition_groups.append(flatten(fp32_avg_grads))
                single_grad_partition_groups.append(fp32_avg_grads)           
                    
            # compute norm   
            param_group = real_master_params[group_id]
            working_grads = real_master_grads[group_id]        
            
            # InternEvo            
            group_name = self.param_groups[group_id]["name"] if "name" in self.param_groups[group_id] else "default"
            group_name = f"{group_id}_{group_name}"
            total_norms[group_name] = self._compute_norm(group_id=group_id, gradients=working_grads, parameters=param_group)
        
            self._grad_store.reset_grads_by_group_id(group_id)

            # update the params in the optimizer
            self.optim.param_groups[group_id]["params"] = real_master_params[group_id]
            
        # update param for moe ep
        # move grad to master param and compute norm
        if len(self.working_moe_params) > 0:
            assert False, "moe"
            moe_grads = []
            for master_moe_param, working_moe_param in zip(self.master_moe_params, self.working_moe_params):
                if master_moe_param.grad is not None:
                    raise RuntimeError("Moe param should not have grad here")
                grad = working_moe_param.grad
                # no need to copy fp32 grad if master_weights is False
                if self._master_weights:
                    grad = grad.to(master_moe_param.dtype).to(master_moe_param.device)
                master_moe_param.grad = grad
                working_moe_param.grad = None
                moe_grads.append(grad)
                grad_partition_groups.append(grad)
            norm_group = self._compute_grad_norm(gradients=moe_grads)
            norm_groups.append(norm_group)
            self.optim.param_groups[-1]["params"] = self.master_moe_params
            del moe_grads
            
        
        found_inf = False
        found_nan = False
        
        if -1 in total_norms.values():
            found_inf = True

        if -2 in total_norms.values():
            found_nan = True
        
        if gpc.config.model.dtype is not torch.float32:
            self.grad_scaler.update(found_inf)
        
        # update loss scale if overflow occurs
        if found_inf:
            if gpc.is_rank_for_log():
                logger.warning("Overflow occurs, please check it.")
                send_alert_message(
                    address=gpc.config.monitor.alert.feishu_alert_address,
                    message="Overflow occurs, please check it.",
                )
            self._grad_store._grads_of_params = dict()
            self.zero_grad()
            return False, total_norms

        if found_nan:
            if gpc.is_rank_for_log():
                logger.warning("Nan grad norm occurs, please check it.")
                send_alert_message(
                    address=gpc.config.monitor.alert.feishu_alert_address,
                    message="Nan grad norm  occurs, please check it.",
                )
            self._grad_store._grads_of_params = dict()
            self.zero_grad()
            return False, total_norms
        
         
        global_norm_groups = {}
        norm_groups = []
        if self._clip_grad_norm > 0:
            for group_name, norm in total_norms.items():
                global_norm_groups[group_name] = norm**0.5
                norm_groups.append(norm**0.5)
                
        
        # collossalAI
        # unscale and clip grads
        global_norm = calculate_global_norm_from_list(norm_list=norm_groups)
        self._unscale_and_clip_grads(grad_partition_groups, global_norm)
        
        # # InternEvo
        # self._unscale_and_clip_grads(single_grad_partition_groups, list(global_norm_groups.values()))
        

        # update the parameters
        self.optim.step()

        # release moe grad
        if len(self.working_moe_params) > 0:
            for master_moe_param, working_moe_param in zip(self.master_moe_params, self.working_moe_params):
                master_moe_param.grad = None
                working_moe_param.data = (
                    master_moe_param.data.to(working_moe_param.device).to(working_moe_param.dtype).detach()
                )

        # release the grad
        grad_partition_groups = []
        for group_id in range(self.num_param_groups):
            release_param_grad(self._master_param_groups_of_current_rank[group_id])

        # update working partition updated by the current rank
        device = get_current_device()
        for group_id in range(self.num_param_groups):
            master_working_param = self.optim.param_groups[group_id]["params"]
            for idx, splited_param in enumerate(master_working_param):
                working_param = real_working_params[group_id][idx]
                if self.moe_extra_dp_pg is not None and False:
                    all_splited_param = [
                        torch.zeros(splited_param.shape, device=device, dtype=self._dtype)
                        for _ in range(self.moe_extra_dp_pg_size)
                    ]
                    dist.all_gather(
                        all_splited_param,
                        splited_param.to(device).to(self._dtype),
                        group=self.moe_extra_dp_pg,
                    )
                else:
                    all_splited_param = [
                        torch.zeros(splited_param.shape, device=device, dtype=self._dtype)
                        for _ in range(self._zero_world_size[group_id])
                    ]
                    dist.all_gather(
                        all_splited_param,
                        splited_param.to(device).to(self._dtype),
                        group=gpc.get_group(self._broadcast_parallel_mode[group_id]),
                    )
                working_param.data.copy_(flatten(all_splited_param)[: working_param.numel()].reshape_as(working_param))
            self.optim.param_groups[group_id]["params"] = self._master_param_groups_of_current_rank[group_id]

        for group_name, global_norm in global_norm_groups.items():
            global_norm_groups[group_name] = global_norm / self.loss_scale
        
        return True, global_norm_groups
    
    def _compute_norm(self, group_id, gradients, parameters):
        # compute norm for gradients that have been reduced
        params = parameters
        grads = gradients

        if len(params) == 0:
            return 0

        norm = 0
        if self._clip_grad_norm > 0:
            # this norm is before scaling, it will be very large
            norm = compute_norm(gradients=grads, parameters=params, zero_mode=self._broadcast_parallel_mode[group_id])

        return norm
        
    #############################
    # Mixed Precision Utilities #
    #############################

    def _unscale_and_clip_grads(self, grad_groups_flat, total_norm_groups):
        
        # colossalAI
        # compute combined scale factor for this group
        
        div_scale = float(self.loss_scale)
        if self._clip_grad_norm > 0.0:
            # norm is in fact norm*scale
            clip = ((total_norm_groups / div_scale) + 1e-6) / self._clip_grad_norm
            if clip > 1:
                div_scale = clip * div_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1.0 / div_scale)
        
        
        # InternEvo        
        # combined_scale_groups = []
        # div_scale = float(self.loss_scale)

        # if self._clip_grad_norm > 0.0:
        #     # norm is in fact norm*scale
        #     for group_id, total_norm in enumerate(total_norm_groups):
        #         combined_scale_groups.append(div_scale)
        #         clip = ((total_norm / div_scale) + 1e-6) / self._clip_grad_norm
        #         if clip > 1:
        #             combined_scale_groups[group_id] = clip * div_scale
       
        # if gpc.is_rank_for_log():
        #     torch.save(grad_groups_flat[0], "grad_groups_flat.pt")
                    
        # for group_id, grads in enumerate(grad_groups_flat):
        #     for grad in grads:
        #         grad.data.mul_(1.0 / combined_scale_groups[group_id])
            
        

    ############################
    # Gradient Synchronization #
    ############################

    # this method is used to sync gradient manually
    def _sync_grad(self):
        for group_id in range(self.num_param_groups):
            param_group = self._working_param_groups[group_id]
            for param in param_group:
                if param.requires_grad and param.grad is not None:
                    self._add_to_bucket(param, group_id)
     
        self._run_reduction()

    def _reduce_grad(self, partition_grad):
        # if not overlapping communication (no reduction hook is attached) when zero1
        # we need to manually reduce these gradients
        if not partition_grad and not self._overlap_communication:
            self._sync_grad()
        else:
            self._run_reduction()

    # this context comes from pytorch DDP
    @contextmanager
    def no_sync(self):
        old_require_grad_sync = self.require_grad_sync
        self.require_grad_sync = False
        try:
            yield
        finally:
            self.require_grad_sync = old_require_grad_sync

    ##############
    # State Dict #
    ##############

    def _pack_state(self, state: Dict) -> Dict:
        # comes from pytorch optimizer.state_dict()
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {id(p): i for i, p in enumerate(group["params"], start_index) if id(p) not in param_mappings}
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.optim.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v for k, v in state.items()}

        return {"state": packed_state, "param_groups": param_groups}

    def state_dict(self) -> Dict:
        """Return a state_dict same with DDP

        Returns:
            Dict: the pytorch form state_dict
        """
        zero_state = dict()
        device = get_current_device()
        for param, state in self.optim.state.items():
            zero_state[param] = copy.deepcopy(state)
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    working_param = self._param_store.master_to_working_param[id(param)]
                    if self.moe_extra_dp_pg is not None and False:
                        gather_tensor = [
                            torch.zeros(v.shape, device=device, dtype=v.dtype) for _ in range(self.moe_extra_dp_pg_size)
                        ]
                        dist.all_gather(gather_tensor, v.to(device), group=self.moe_extra_dp_pg)
                    else:
                        pass
                        # gather_tensor = [
                        #     torch.zeros(v.shape, device=device, dtype=v.dtype) for _ in range(self._zero_world_size[group_id])
                        # ]
                    #     dist.all_gather(gather_tensor, v.to(device), group=gpc.get_group(self._broadcast_parallel_mode[group_id]))
                    # param_state = (
                    #     torch.stack(gather_tensor).view(-1)[: working_param.numel()].reshape_as(working_param).cpu()
                    # )
                    # zero_state[param][k] = param_state

        states_dict = self._pack_state(zero_state)

        return states_dict

    def load_state_dict(self, state_dict: Dict):
        """Load state dict, requires the state_dict be the pytorch form

        Args:
            state_dict (dict): A pytorch form state_dict
        """
        zero_state_dict = copy.deepcopy(state_dict)
        for param_idx, state in zero_state_dict["state"].items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    pass
                    # padding_size = (self._zero_world_size[group_id] - v.numel() % self._zero_world_size[group_id]) % self._zero_world_size[group_id]
                    # with torch.no_grad():
                    #     v = v.flatten()
                    #     if padding_size > 0:
                    #         v = torch.nn.functional.pad(v, [0, padding_size])
                    #     if self.moe_extra_dp_pg is not None and False:
                    #         v_list = v.split(v.numel() // self.moe_extra_dp_pg_size)
                    #         zero_state_dict["state"][param_idx][k] = v_list[self.moe_extra_dp_pg_rank].detach().clone()
                    #     else:
                    #         v_list = v.split(v.numel() // self._zero_world_size[group_id])
                            ### todo
                            # zero_state_dict["state"][param_idx][k] = v_list[self._zero_local_rank[group_id]].detach().clone()

        self.optim.load_state_dict(zero_state_dict)

    def state_dict_shard(self, max_shard_size: int = 1024) -> Iterator[Tuple[Dict, int]]:
        """Returns dictionaries containing a whole state of the module one by one. The max size of dictionary shard is specified by ``max_shard_size``.
           Only include the 'state' in state_dict.

        Args:
            max_shard_size (int, optional): max size of state shard (in MB). Defaults to 1024.

        Yields:
            Iterator[OrderedDict]: A generator of state dict shard
        """
        ret_block = dict()
        ret_block_size = 0

        device = get_current_device()
        local_states = self.optim.state_dict()["state"]
        for param_idx, states in local_states.items():
            current_block_size = 0
            current_block = copy.deepcopy(states)

            # find the working param of current param_id
            for group_id, pg in self._master_param_groups_of_current_rank.items():
                if (group_id + 1) * len(pg) < param_idx:
                    continue
                master_param = pg[param_idx - (group_id) * len(pg)]
                working_param = self._param_store.master_to_working_param[id(master_param)]

            for k, v in states.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    if self.moe_extra_dp_pg is not None and False:
                        state_tensor = [
                            torch.zeros(v.shape, device=device, dtype=v.dtype) for _ in range(self.moe_extra_dp_pg_size)
                        ]
                        dist.all_gather(state_tensor, v.to(device), group=self.moe_extra_dp_pg)
                    else:
                        state_tensor = [
                            torch.zeros(v.shape, device=device, dtype=v.dtype) for _ in range(self._zero_world_size[group_id])
                        ]
                        dist.all_gather(state_tensor, v.to(device), group=gpc.get_group(self._broadcast_parallel_mode[group_id]))
                    state_tensor = (
                        torch.stack(state_tensor).view(-1)[: working_param.numel()].reshape_as(working_param).cpu()
                    )
                    current_block_size += state_tensor.numel()
                    current_block[k] = state_tensor

            if ret_block_size + current_block_size > max_shard_size and len(ret_block) > 0:
                yield ret_block, ret_block_size
                ret_block = dict()
                ret_block_size = 0

            ret_block[param_idx] = current_block
            ret_block_size += current_block_size

        yield ret_block, ret_block_size

    def update_master_params(self, model: nn.Module) -> None:
        """Update master params from working params

        Args:
            model (nn.Module): The model to update master params
        """
        for p in model.parameters():
            p_id = id(p)
            if p_id in self._param_store.working_to_master_param:
                master_param = self._param_store.working_to_master_param[p_id]
                padding_size = self._param_store.get_param_padding_size(p)
                working_param = p.data.view(-1)
                if padding_size > 0:
                    working_param = torch.nn.functional.pad(working_param, [0, padding_size])
                if self.moe_extra_dp_pg is not None and False:
                    master_param.copy_(working_param.chunk(self.extra_dp_pg_size)[self.extra_dp_pg_rank])
                else:
                    ### todo
                    pass
                    # master_param.copy_(working_param.chunk(self._zero_world_size[group_id])[self._zero_local_rank[group_id]])
        if hasattr(self, "master_moe_params"):
            for master_moe_param, working_moe_param in zip(self.master_moe_params, self.working_moe_params):
                master_moe_param.copy_(working_moe_param)

    def get_working_to_master_map(self) -> Dict[int, torch.Tensor]:
        return self._param_store.working_to_master_param

    def get_master_to_working_map(self) -> Dict[int, torch.Tensor]:
        if hasattr(self, "moe_master_to_working_map"):
            return {
                **self._param_store.master_to_working_param,
                **self.moe_master_to_working_map,
            }
        return self._param_store.master_to_working_param
    
    def _attach_reduction_hook(self):
        # we iterate over the fp16 params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._working_param_groups[group_id]
            for param in param_group:
                # we should not reduce the param in moe
                if not param.requires_grad:
                    continue

                reduce_rank = None

                def _define_and_attach(param, reduce_rank=None):
                    
                    def grad_handler(group_id, param):
                        # if run with no_sync context, would not sync grad when backward
                        if self.require_grad_sync:
                            self._add_to_bucket(param, group_id)
                        
                    reduce_scatter_checker = partial(
                        self._wait_reduce_scatter_and_accumulate_grads,
                        param=param,
                    )

                    def reduction_layernorm_func():
                        handle = reduce_tensor(
                            param.grad,
                            dtype=None,
                            dst_rank=reduce_rank,
                            parallel_mode=ParallelMode.WEIGHT if self.use_isp else ParallelMode.TENSOR,
                        )
                        handle.wait()

                    # define hook for real gradient accumulation.
                    
                    def accum_grad_hook(*args):  # pylint: disable=W0613
                        reduce_scatter_checker()

                    # define hook for sequence_parallel
                    def extra_layernorm_reduce_grad_hook(*args):  # pylint: disable=W0613
                        if self.skip_grad_reduce is False:
                            reduction_layernorm_func()


                    # the grad of layernorm should be all-reduce across the global process group
                    # here is the first stage all-reduce in tp/wp process group
                    # the second stage all-reduce will be processed in reduce_grad_hook
                    if (
                        is_using_sequence_parallel()
                        and hasattr(param, IS_REPLICA_ZERO_PARALLEL)
                        and getattr(param, IS_REPLICA_ZERO_PARALLEL) is True
                    ):
                        param.register_post_accumulate_grad_hook(extra_layernorm_reduce_grad_hook)

                    # we should not only register for parameters which have isp_reduce_scatter_name attr.
                    # we must keep up with reduce_grad_hook.
                    if (
                        self._isp_communicator
                        and self._isp_communicator.overlap
                        and gpc.config.parallel.weight.size > 1
                    ):
                        param.register_post_accumulate_grad_hook(accum_grad_hook)
                        
                    if self._overlap_sync_grad:
                        param.register_post_accumulate_grad_hook(partial(grad_handler, group_id))          

                _define_and_attach(param, reduce_rank)


### todo
from .base_optimizer import BaseOptimizer
def reload_zero_fp32_buff(optimizer):
    # If we use AMP optimizer, we need to update its fp32 buffer as newly loaded weights value.
    # Or we must ensure that loading model weights must be done before zero is initialized.
    if isinstance(optimizer, HybridZeroOptimizer_v2):
        for group_id, param_group in enumerate(optimizer.optim.param_groups):
            if optimizer.param_group_has_params[group_id]:
                # flatten fp16 params have already been updated by 'load_model_checkpoint'
                fp16_flat_current_rank = optimizer._param_store.get_flat_fp16_param_by_rank_group(
                    optimizer._zero_local_rank[group_id], group_id
                )
                # param_group["params"] is fp32 flatten optimizer states of this zero rank.
                param_group["params"][0].data.copy_(fp16_flat_current_rank.float())