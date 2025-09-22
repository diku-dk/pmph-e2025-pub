-- Primes: a Nested Version but with sequential loops, so
--         please use the C backend for running, because
--         nothing is parallel.
-- Since Futhark does not support irregular array,
-- we pad to the maximal allowed dimension.
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out

let primesSeq (n: i64) : []i64 =
  let sq_primes = [2, 3]
  let len = 4
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i32 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\p -> len / p - 1) sq_primes

      ----------------------------------------------------------------
      ----------------------------------------------------------------
      -- the current iteration knowns the primes <= 'len', 
      -- based on which it will compute the primes <= 'len*len' 
      -- Morally, the code should be the nested-parallel code above,
           -- composite = map (\ p -> let m = (n `div` p)
           --                      in  map (\ j -> j * p ) (drop 2 (iota (m+1)))
           --                 ) sq_primes
           -- not_primes = reduce (++) [] composite
      -- but since Futhark does not support irregular arrays
      -- we write it as a loop nest in which we precompute
      -- the total result size
      --
      let flat_size = reduce (+) 0 mult_lens
      let not_primes = replicate flat_size 0
      let cur_ind = 0
      let (not_primes, _) = 
        loop (not_primes, cur_ind) for i < length sq_primes do
            let p = sq_primes[i]
            let s = mult_lens[i]
            let not_primes = 
            loop not_primes for j < s do
              not_primes with [cur_ind + j] = (j + 2) * p
            in (not_primes, cur_ind + s)
       -------------------------------------------------------------
       -------------------------------------------------------------

       let zero_array  = replicate flat_size 0
       let mostly_ones = map (i32.bool <-< (> 1)) (iota (len + 1))
       let prime_flags = scatter mostly_ones not_primes zero_array
       let sq_primes   = filter (\i-> #[unsafe] prime_flags[i] > 0i32) (0...len)
       in  (sq_primes, len)
  in sq_primes

let main (n: i64) : []i64 = primesSeq n
