--------------------------------------------------------
--- Attempting the sequential implementation of lssp ---
--------------------------------------------------------

type int = i32
let max (x:int, y:int) = i32.max x y

let lssp_seq [n]
             (pred1 : int -> bool)
             (pred2 : int -> int -> bool) 
             (xs    : [n]int ) : int =
  let (best_len, _, _) =
  loop(best_len, curr_len, prev) = (0, 0, 0)
    for i < n do
        let x = xs[i]
        let connected = if i==0 then true else pred2 prev x
        let satisfied = pred1 x
        let curr_len  = if satisfied && connected
                        then curr_len+1
                        else if satisfied then 1 else 0
        let best_len = max(best_len, curr_len)
        in  (best_len, curr_len, x)
  in best_len
