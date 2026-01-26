%% viva_tensor_ffi.erl - O(1) array operations via Erlang :array
-module(viva_tensor_ffi).
-export([
    list_to_array/1,
    array_to_list/1,
    array_get/2,
    array_set/3,
    array_size/1,
    array_dot/2,
    array_matmul/5,
    array_sum/1,
    array_scale/2,
    strided_get/4,
    send_msg/2,
    collect_n/1,
    receive_any/0
]).

%% Convert list to array (O(n) once, then O(1) access)
list_to_array(List) ->
    array:from_list(List).

%% Convert array back to list
array_to_list(Array) ->
    array:to_list(Array).

%% O(1) random access
array_get(Array, Index) ->
    array:get(Index, Array).

%% O(1) functional update (returns new array)
array_set(Array, Index, Value) ->
    array:set(Index, Value, Array).

%% Array size
array_size(Array) ->
    array:size(Array).

%% Dot product
array_dot(A, B) ->
    Size = array:size(A),
    dot_loop(A, B, 0, Size, 0.0).

dot_loop(_A, _B, Idx, Size, Acc) when Idx >= Size -> Acc;
dot_loop(A, B, Idx, Size, Acc) ->
    Val = array:get(Idx, A) * array:get(Idx, B),
    dot_loop(A, B, Idx + 1, Size, Acc + Val).

%% Matrix multiplication
%% matmul(A, B, M, N, K) where A is MxK, B is KxN -> Result MxN
array_matmul(A, B, M, N, K) ->
    Result = [
        begin
            RowStart = I * K,
            lists:foldl(fun(KIdx, Acc) ->
                AVal = array:get(RowStart + KIdx, A),
                BVal = array:get(KIdx * N + J, B),
                Acc + AVal * BVal
            end, 0.0, lists:seq(0, K - 1))
        end
        || I <- lists:seq(0, M - 1),
           J <- lists:seq(0, N - 1)
    ],
    array:from_list(Result).

%% Sum all elements in array
array_sum(Array) ->
    array:foldl(fun(_Idx, Val, Acc) -> Acc + Val end, 0.0, Array).

%% Scale all elements by scalar
array_scale(Array, Scalar) ->
    array:map(fun(_Idx, Val) -> Val * Scalar end, Array).

%% Strided access - NumPy-style indexing
%% Given strides [s0, s1, ...] and indices [i0, i1, ...], compute:
%% offset + i0*s0 + i1*s1 + ...
strided_get(Array, Offset, Strides, Indices) ->
    FlatIdx = compute_strided_index(Offset, Strides, Indices),
    array:get(FlatIdx, Array).

compute_strided_index(Offset, Strides, Indices) ->
    lists:foldl(
        fun({Stride, Idx}, Acc) -> Acc + Stride * Idx end,
        Offset,
        lists:zip(Strides, Indices)
    ).

%% Send message to pid (for concurrent benchmarks)
send_msg(Pid, Msg) ->
    Pid ! Msg,
    Msg.

%% Collect N messages from mailbox
collect_n(N) ->
    collect_n(N, []).

collect_n(0, Acc) ->
    lists:reverse(Acc);
collect_n(N, Acc) ->
    receive
        Msg -> collect_n(N - 1, [Msg | Acc])
    end.

%% Receive any message (blocking)
receive_any() ->
    receive
        Msg -> Msg
    end.
