-- Primes: Naive Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out

let primesNaive (n: i64) : []i64 =
  let a = map (i32.bool <-< (>= 2)) (iota (n + 1))
  let sqrt_n = i64.f64 (f64.sqrt (f64.i64 n))
  let primes_flags =
    loop a for i < sqrt_n - 1 do
      let i = i + 2 in
      if a[i] == 0i32 then a
      else
        let m    = (n / i) - 1
        let inds = map ((*i) <-< (+2)) (iota m)
        let vals = replicate m 0
        in scatter a inds vals
  in filter (\i -> #[unsafe] primes_flags[i] != 0) (iota (n + 1))


-- Simplest way is to use 
--   $ futhark bench --backend=cuda primes-naive.fut
-- You may also compile
--   $ futhark cuda primes-naive.fut
-- and run with:
--   $ echo "10000000i64" | ./primes-naive -t /dev/stderr -r 10 > /dev/null

let main (n: i64) : []i64 = primesNaive n
