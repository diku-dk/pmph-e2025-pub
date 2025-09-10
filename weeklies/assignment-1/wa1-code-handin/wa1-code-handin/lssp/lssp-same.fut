-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }

-- ==
-- compiled input { [7i32] }
-- output { 1 }

-- ==
-- compiled input { [1i32,1,1,2,2,2,2,3] }
-- output { 4 }

-- ==
-- compiled input { [3i32,3,2,2,2,1,1,1,1,0] }
-- output { 4 }

import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

entry bench_same_gpu (n: i32) : i32 =
  let pred1 _   = true
  let pred2 x y = (x == y)
  let xs = map (\i -> i32.i64 ((i / 5i64) % 97i64))
               (iota (i64.i32 n))
  in lssp pred1 pred2 xs

entry bench_same_seq (n: i32) : i32 =
  let pred1 _   = true
  let pred2 x y = (x == y)
  let xs = map (\i -> i32.i64 ((i / 5i64) % 97i64))
               (iota (i64.i32 n))
  in lssp_seq pred1 pred2 xs

-- ==
-- entry: bench_same_gpu
-- input { 1000000i32 }
-- input { 5000000i32 }

-- ==
-- entry: bench_same_seq
-- input { 1000000i32 }
-- input { 5000000i32 }
