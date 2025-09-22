-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out

let primesFlat (n: i64) : []i64 =
  let sq_primes = [2i64, 3i64, 5i64, 7i64]
  let len = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\p -> (len / p) - 1) sq_primes
      let flat_size = reduce (+) 0 mult_lens

      --------------------------------------------------------------
      -- The current iteration knows the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the correct `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`. 
      
      -- Flattened construction of all j-values ([2..mm1] per segment)
      -- and replication of corresponding primes; then elementwise multiply.
      let pl_pairs   = zip sq_primes mult_lens
      let nz_pairs   = filter (\(_p,l) -> l > 0) pl_pairs
      let (primes_nz, lens_nz) = unzip nz_pairs

      -- compute segment starts from lengths
      let inc     = scan (+) 0 lens_nz   
      let starts  = map2 (-) inc lens_nz  

      -- flags marking the beginning of each (non-empty) segment
      -- (derive value array from 'starts' so sizes unify for 'scatter')
      let flag_vals = map (\_ -> 1i64) starts
      let flags     = scatter (replicate flat_size 0i64) starts flag_vals

      -- segment id for each flat position, then position-within-segment
      let seg_inc   = scan (+) 0 flags
      let seg_id    = map (\x -> x - 1i64) seg_inc
      let pos       = iota flat_size
      let start_of_pos = map (\sid -> #[unsafe] starts[sid]) seg_id
      let within    = map2 (-) pos start_of_pos
      let j_values  = map (\w -> w + 2i64) within      -- [2..mm1] per segment

      -- replicate primes per segment and form all composites
      let primes_per_pos = map (\sid -> #[unsafe] primes_nz[sid]) seg_id
      let not_primes = map2 (*) primes_per_pos j_values

      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array  = replicate flat_size 0i8
       let mostly_ones = map (i8.bool <-< (> 1)) (iota (len + 1))
       let prime_flags = scatter mostly_ones not_primes zero_array
       let sq_primes   = filter (\i -> #[unsafe] prime_flags[i] != 0) (iota (len + 1))
       in (sq_primes, len)
  in sq_primes

-- RUN a big test with:
--   $ futhark cuda primes-flat.fut
--   $ echo "10000000i64" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
-- or simply use futhark bench, i.e.,
--   $ futhark bench --backend=cuda primes-flat.fut
let main (n: i64) : []i64 = primesFlat n
