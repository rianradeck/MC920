f = open("in.in", 'r', encoding="utf-8")

ans = []
for s in f:    
    ans.append([int(x) for x  in s.split()])

print(ans)