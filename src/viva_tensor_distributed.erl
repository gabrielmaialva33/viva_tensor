%% viva_tensor_distributed.erl - Distributed tensor operations
%%
%% Provides BEAM-native distributed computing for tensor operations.
%% Uses Erlang's built-in distribution for zero-copy message passing.
%%
%% Key concepts:
%% - Row sharding: Split matrix A by rows across nodes
%% - Broadcast B: Send matrix B to all nodes
%% - Parallel execution: Each node computes its shard
%% - Gather results: Collect and combine partial results

-module(viva_tensor_distributed).
-export([
    spawn_matmul_task/6,
    await_task/1,
    connect_nodes/1,
    list_nodes/0,
    shard_tensor/3,
    gather_results/1,
    parallel_matmul/5
]).

%% ==========================================================================
%% Task Spawning
%% ==========================================================================

%% Spawn a matrix multiplication task on a remote node
%% Returns a reference that can be awaited
spawn_matmul_task({node, NodeName}, A, B, M, N, K) ->
    Caller = self(),
    Ref = make_ref(),

    %% Spawn on remote node (or local if same node)
    Node = list_to_atom(binary_to_list(NodeName)),
    spawn(Node, fun() ->
        Result = compute_matmul(A, B, M, N, K),
        Caller ! {matmul_result, Ref, Result}
    end),

    {task_ref, Ref}.

%% Await a task result with timeout
await_task({task_ref, Ref}) ->
    receive
        {matmul_result, Ref, Result} ->
            {ok, Result}
    after 30000 ->
        {error, <<"timeout">>}
    end.

%% ==========================================================================
%% Node Management
%% ==========================================================================

%% Connect to a list of nodes
connect_nodes(NodeNames) ->
    Results = [net_adm:ping(list_to_atom(binary_to_list(N))) || N <- NodeNames],
    {ok, Results}.

%% List all connected nodes
list_nodes() ->
    Nodes = [node() | nodes()],
    [{node, atom_to_binary(N, utf8)} || N <- Nodes].

%% ==========================================================================
%% Tensor Sharding
%% ==========================================================================

%% Shard a tensor (list of floats) into N parts by rows
%% Returns list of {Data, RowCount} tuples
shard_tensor(Data, RowSize, NumShards) ->
    TotalRows = length(Data) div RowSize,
    RowsPerShard = TotalRows div NumShards,
    Remainder = TotalRows rem NumShards,

    shard_tensor_acc(Data, RowSize, RowsPerShard, Remainder, NumShards, 0, []).

shard_tensor_acc(_Data, _RowSize, _RowsPerShard, _Remainder, NumShards, Current, Acc)
  when Current >= NumShards ->
    lists:reverse(Acc);
shard_tensor_acc(Data, RowSize, RowsPerShard, Remainder, NumShards, Current, Acc) ->
    %% First 'Remainder' shards get an extra row
    Extra = case Current < Remainder of true -> 1; false -> 0 end,
    Rows = RowsPerShard + Extra,
    Elements = Rows * RowSize,

    {ShardData, Rest} = lists:split(min(Elements, length(Data)), Data),
    Shard = {ShardData, Rows},

    shard_tensor_acc(Rest, RowSize, RowsPerShard, Remainder, NumShards, Current + 1, [Shard | Acc]).

%% ==========================================================================
%% Gather Results
%% ==========================================================================

%% Gather results from multiple task refs
gather_results(TaskRefs) ->
    gather_results_acc(TaskRefs, []).

gather_results_acc([], Acc) ->
    {ok, lists:append(lists:reverse(Acc))};
gather_results_acc([TaskRef | Rest], Acc) ->
    case await_task(TaskRef) of
        {ok, Result} ->
            gather_results_acc(Rest, [Result | Acc]);
        {error, Reason} ->
            {error, Reason}
    end.

%% ==========================================================================
%% Parallel Matrix Multiplication
%% ==========================================================================

%% High-level API: Parallel matmul across connected nodes
%% A[M,K] @ B[K,N] -> C[M,N]
parallel_matmul(A, B, M, N, K) ->
    Nodes = list_nodes(),
    NumNodes = length(Nodes),

    case NumNodes of
        0 ->
            {error, <<"no_nodes">>};
        1 ->
            %% Single node: just compute locally
            {ok, compute_matmul(A, B, M, N, K)};
        _ ->
            %% Multiple nodes: shard and distribute
            Shards = shard_tensor(A, K, NumNodes),

            %% Spawn tasks on each node
            TaskRefs = lists:zipwith(
                fun(Node, {ShardData, ShardRows}) ->
                    spawn_matmul_task(Node, ShardData, B, ShardRows, N, K)
                end,
                Nodes, Shards
            ),

            %% Gather and combine results
            gather_results(TaskRefs)
    end.

%% ==========================================================================
%% Local Computation (called on each node)
%% ==========================================================================

%% Compute matmul locally using best available backend
compute_matmul(A, B, M, N, K) ->
    %% Try Zig SIMD first, then Accelerate, then pure Erlang
    case viva_tensor_zig:is_loaded() of
        true ->
            case viva_tensor_zig:simd_matmul(A, B, M, N, K) of
                {ok, Result} -> Result;
                _ -> fallback_matmul(A, B, M, N, K)
            end;
        false ->
            case viva_tensor_nif:is_nif_loaded() of
                true ->
                    case viva_tensor_nif:matmul(A, B, M, N, K) of
                        {ok, Result} -> Result;
                        _ -> fallback_matmul(A, B, M, N, K)
                    end;
                false ->
                    fallback_matmul(A, B, M, N, K)
            end
    end.

%% Pure Erlang fallback
fallback_matmul(AList, BList, M, N, K) ->
    A = array:from_list(AList),
    B = array:from_list(BList),
    Result = viva_tensor_ffi:array_matmul(A, B, M, N, K),
    array:to_list(Result).
