-- Primes: Adhoc Version
-- ==
-- compiled input { 30i64 } output { [2i64,3i64,5i64,7i64,11i64,13i64,17i64,19i64,23i64,29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out

-- run with: futhark bench --backend=cuda primes-adhoc.fut
-- or by compiling with
-- futhark cuda primes-adhoc.fut
-- echo "10000000i64" | ./primes-adhoc -t /dev/stderr > /dev/null

-- Find the first n primes
let primesAdHoc (n: i64) : []i64 =
  let (acc, _) =
    loop (acc, c) = ([], 2i64)
      while c < n do
        let c2 = if n / c < c then n else c*c -- i32.min (c * c) (n+1)
        let is = map (+c) (iota (c2 - c))
        let fs = map (\i ->
                        map (\p -> i32.bool (i % p == 0)) acc
                        |> reduce (+) 0
                     ) is
        -- apply the sieve
        let new = filter (\i -> #[unsafe] fs[i - c] == 0) is
        in (acc ++ new, c2)
  in acc

-- Return the number of primes less than n
let main (n: i64) : []i64 =
  primesAdHoc n
