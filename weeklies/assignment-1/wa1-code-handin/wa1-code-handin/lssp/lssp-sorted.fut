-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }  
-- output { 
--    9
-- }

-- ==
-- compiled input { [5i32] }
-- output { 1 }

-- ==
-- compiled input { [1i32, 2, 2, 3, 0, 1, 2] }
-- output { 4 }

-- ==
-- compiled input { [3i32,2,1,0] }
-- output { 1 }

import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

entry bench_sorted_gpu (n: i32) : i32 =
  let pred1 _   = true
  let pred2 x y = (x <= y)
  let xs = map (\i -> i32.i64 (i % 1000i64))
               (iota (i64.i32 n))
  in lssp pred1 pred2 xs

entry bench_sorted_seq (n: i32) : i32 =
  let pred1 _   = true
  let pred2 x y = (x <= y)
  let xs = map (\i -> i32.i64 (i % 1000i64))
               (iota (i64.i32 n))
  in lssp_seq pred1 pred2 xs

-- ==
-- entry: bench_sorted_gpu
-- input { 1000000i32 }
-- input { 5000000i32 }

-- ==
-- entry: bench_sorted_seq
-- input { 1000000i32 }
-- input { 5000000i32 }
