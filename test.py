import id3

varnames = ["first", "second", "output"]

x = [
        [1,0,1],
        [1,0,0],
        [1,1,0],
        [1,1,1],
        [0,0,0],
        [1,0,1],
        [1,0,1],
        [1,0,1],
        [1,0,1],
        [1,0,1],
        [1,0,1],
        [1,0,1],
    ]

print(id3.partition_on_attr(x,1))
