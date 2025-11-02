-- ASSIGNMENT 1: Flat-Parallel implementation of Sparse Matrix-Vector Multiplication
-- ==
-- compiled input {
--   [0i64, 1i64, 0i64, 1i64, 2i64, 1i64, 2i64, 3i64, 2i64, 3i64, 3i64]
--   [2.0f32, -1.0f32, -1.0f32, 2.0f32, -1.0f32, -1.0f32, 2.0f32, -1.0f32, -1.0f32, 2.0f32, 3.0f32]
--   [2i64, 3i64, 3i64, 2i64, 1i64]
--   [2.0f32, 1.0f32, 0.0f32, 3.0f32]
-- }
-- output { [3.0f32, 0.0f32, -4.0f32, 6.0f32, 9.0f32] }

------------------------
--- Sgm Scan Helpers ---
------------------------
-- Generic segmented scan (generic in the binary operator and in the element
-- type, t, of the segmented array).
let sgmScan [n] 't
            (op: t -> t -> t)
            (ne: t)
            (flags: [n]bool)
            (vals: [n]t)
            : [n]t =
  scan (\(f1, v1) (f2, v2) -> (f1 || f2, if f2 then v2 else op v1 v2))
       (false, ne)
       (zip flags vals)
  |> unzip
  |> (.1)

-- Segmented cumulative summation over 32-bit float segmented arrays.
let sgmSumF32 [n] (flags: [n]bool) (vals: [n]f32) : [n]f32 =
  sgmScan (+) 0f32 flags vals

-----------------------------------------------------
-- Please implement the function below, currently dummy,
-- which is supposed to implement sparse-matrix vector
-- multiplication. Note that the shp array contains the
-- sizes of each row of the matrix.
-----------------------------------------------------
---  Dense Matrix:                                ---
---  [ 2.0, -1.0,  0.0, 0.0]                      ---
---  [-1.0,  2.0, -1.0, 0.0]                      ---
---  [ 0.0, -1.0,  2.0,-1.0]                      ---
---  [ 0.0,  0.0, -1.0, 2.0]                      ---
---  [ 0.0,  0.0,  0.0, 3.0]                      ---
---                                               ---
---  In the sparse, nested-parallel format it is  ---
---  represented as a list of lists named mat     ---
---  [ [(0,2.0),  (1,-1.0)],                      ---
---    [(0,-1.0), (1, 2.0), (2,-1.0)],            ---
---    [(1,-1.0), (2, 2.0), (3,-1.0)],            ---
---    [(2,-1.0), (3, 2.0)],                      ---
---    [(3,3.0)]                                  ---
---  ]                                            ---
---                                               ---
--- The nested-parallel code is something like:   ---
--- map (\ row ->                                 ---
---         let prods =                           ---
---               map (\(i,x) -> x*vct[i]) row    ---
---         in  reduce (+) 0 prods                ---
---     ) mat                                     ---
---                                               ---
--- mat is the flattened data of the matrix above,---
---  while shp holds the sizes of each row, i.e., ---
---                                               ---
--- mat_val =
---       [ (0,2.0),(1,-1.0),(0,-1.0),(1, 2.0),   ---
---         (2,-1.0),(1,-1.0),(2, 2.0),(3,-1.0),  ---
---         (2,-1.0),(3, 2.0),(3,3.0)             ---
---       ]                                       ---
--- mat_shp = [2,3,3,2,1]                         ---
---
--- The vector is dense and matches the number of ---
---   columns of the matrix                       ---
---   e.g., x = [2.0, 1.0, 0.0, 3.0] (transposed) ---
---                                               ---
--- YOUR TASK is to implement the function below  ---
--- such that it consists of only flat-parallel   ---
--- operations and is semantically equivalent to  ---
--- the nested parallel program described above   ---
--- See also Section 3.2.4 ``Sparse-Matrix Vector ---
--- Multiplication'' in lecture notes, page 40-41.---
--- You may use in your implementation the        ---
--- `sgmSumF32` function provided above.          ---
--- You may also take a look at the sequential    ---
--- implementation in file spMVmult-seq.fut.      ---
--- If the futhark-opencl compiler complains about---
---  unsafe code try wrapping the offending       ---
---  expression with keyword `unsafe`.            ---
---                                               ---
--- Necessary steps for the flat-parallel implem: ---
--- 1. you need to compute the flag array from shp---
---    if you cannot figure it out, take a look   ---
---    at `mkFlagArray` in lecture notes,         ---
---    section 4, page 48                         ---
--- 2. you need to multiply all elements of the   ---
---    matrix with their corresponding vector     ---
---    element                                    ---
--- 3. you need to sum up the products above      ---
---    across each row of the matrix. This can    ---
---    be achieved with a segmented scan and then ---
---    with a map that extracts the last element  ---
---    of the segment.
-----------------------------------------------------
let spMatVctMult [num_elms][vct_len][num_rows]
                 (mat_val: [num_elms](i64, f32))
                 (mat_shp: [num_rows]i64)
                 (vct: [vct_len]f32)
                   : [num_rows]f32 =

  let shp_sc = scan (+) 0 mat_shp
  -- TODO: fill in your implementation here.
  --       for now, the function simply returns zeroes.
  let starts  : [num_rows]i64  = map2 (-) shp_sc mat_shp
  
  let flags   : [num_elms]bool = scatter (replicate num_elms false)
                                     starts (replicate num_rows true)
                                     
  let prods   : [num_elms]f32  = map (\(i,x) -> x * vct[i]) mat_val

  let partial : [num_elms]f32  = sgmSumF32 flags prods

  let ends    : [num_rows]i64  = map (\s -> s - 1i64) shp_sc

  in map2 (\len e -> if len == 0i64 then 0.0f32 else partial[e]) mat_shp ends


-- One may run with for example:
-- $ futhark dataset --i64-bounds=0:9999 -g [1000000]i64 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i64-bounds=100:100 -g [10000]i64 --f32-bounds=-10.0:10.0 -g [10000]f32 | ./spMVmult-seq -t /dev/stderr -n
let main [n][m]
         (mat_inds: [n]i64)
         (mat_vals: [n]f32)
         (shp: [m]i64)
         (vct: []f32)
           : [m]f32 =
  spMatVctMult (zip mat_inds mat_vals) shp vct

-- ==
-- compiled input {
--   [0i64, 1i64, 0i64, 1i64, 2i64, 1i64, 2i64, 3i64, 2i64, 3i64, 3i64]
--   [2.0f32, -1.0f32, -1.0f32, 2.0f32, -1.0f32, -1.0f32, 2.0f32, -1.0f32, -1.0f32, 2.0f32, 3.0f32]
--   [2i64, 3i64, 3i64, 2i64, 1i64]
--   [2.0f32, 1.0f32, 0.0f32, 3.0f32]
-- }
-- output { [3.0f32, 0.0f32, -4.0f32, 6.0f32, 9.0f32] }

-- ==
-- compiled input {
--   [0i64, 1i64, 0i64, 1i64, 2i64, 1i64, 2i64, 3i64, 2i64, 3i64, 3i64]
--   [2.0f32, -1.0f32, -1.0f32, 2.0f32, -1.0f32, -1.0f32, 2.0f32, -1.0f32, -1.0f32, 2.0f32, 3.0f32]
--   [2i64, 3i64, 3i64, 2i64, 1i64]
--   [1.0f32, 1.0f32, 1.0f32, 1.0f32]
-- }
-- output { [1.0f32, 0.0f32, 0.0f32, 1.0f32, 3.0f32] }

-- ==
-- compiled input {
--   [0i64, 2i64, 1i64]
--   [5.0f32, -1.0f32, 0.5f32]
--   [1i64, 1i64, 1i64]
--   [1.0f32, 2.0f32, 4.0f32]
-- }
-- output { [5.0f32, -4.0f32, 1.0f32] }

-- ==
-- compiled input {
--   [0i64, 2i64, 1i64,  2i64, 4i64]
--   [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]
--   [3i64, 0i64, 2i64]
--   [10.0f32, 20.0f32, 30.0f32, 40.0f32, 50.0f32]
-- }
-- output { [130.0f32, 0.0f32, 370.0f32] }