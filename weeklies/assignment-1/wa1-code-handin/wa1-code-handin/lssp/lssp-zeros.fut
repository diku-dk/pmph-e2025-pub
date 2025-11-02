-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }

-- ==
-- compiled input { [1i32,2,3] }
-- output { 0 }

-- ==
-- compiled input { [0i32] }
-- output { 1 }

-- ==
-- compiled input { [0i32,0,1,0,0,0,2,0] }
-- output { 3 }

import "lssp-seq"
import "lssp"

type int = i32

let main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

-- Bench entries (large synthetic inputs)
entry bench_zeros_gpu (n: i32) : i32 =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
  let xs = map (\i -> if (i % 8i64 == 0i64) then 1i32 else 0i32)
               (iota (i64.i32 n))
  in lssp pred1 pred2 xs

entry bench_zeros_seq (n: i32) : i32 =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
  let xs = map (\i -> if (i % 8i64 == 0i64) then 1i32 else 0i32)
               (iota (i64.i32 n))
  in lssp_seq pred1 pred2 xs

-- Bench datasets
-- ==
-- entry: bench_zeros_gpu
-- input { 1000000i32 }
-- input { 5000000i32 }

-- ==
-- entry: bench_zeros_seq
-- input { 1000000i32 }
-- input { 5000000i32 }

