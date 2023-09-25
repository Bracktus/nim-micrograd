import math
import strformat
import sets
import hashes
import algorithm

proc doNothing(): void =
    discard nil

type 
    Value* = ref object
        data*: float
        grad*: float = 0
        label*: string = ""

        prev: seq[Value] = @[]
        diff: proc(): void = doNothing


proc `$`*(self: Value): string =
    fmt"Value(data: {self.data}, grad: {self.grad}, label: {self.label})"


proc `+`*(self: Value, other: Value): Value =
    let newVal = Value(
        data: self.data + other.data, 
        prev: @[self, other],
    )

    proc diff(): void =
        self.grad += newVal.grad
        other.grad += newVal.grad

    newVal.diff = diff
    newVal

proc `+`*(self: Value, other: float): Value =
    self + Value(data: other, prev: @[])

proc `+`*(other: float, self: Value): Value =
    self + Value(data: other, prev: @[])

proc `*`*(self: Value, other: Value): Value =
    let newVal = Value(
        data: self.data * other.data,
        prev: @[self, other],
    )

    proc diff(): void =
        self.grad += other.data * newVal.grad 
        other.grad += self.data * newVal.grad 

    newVal.diff = diff
    newVal

proc `*`*(self: Value, other: float): Value =
    self * Value(data: other, prev: @[]) 

proc `*`*(other: float, self: Value): Value =
    self * Value(data: other, prev: @[])

proc pow*(self: Value, other: float): Value =
    let newVal = Value(
        data: pow(self.data, other),
        prev: @[self],
    ) 

    proc diff(): void =
        self.grad += other * pow(self.data, other - 1) * newVal.grad

    newVal.diff = diff
    newVal

proc `-`*(self: Value): Value =
    self * -1.0

proc `-`*(self: Value, other: Value | float): Value = 
    self + -other

proc `-`*(other: float, self: Value): Value = 
    -self + other

proc `/`*(self: Value, other: Value): Value =
    self * pow(other, -1)

proc `/`*(self: Value, other: float): Value =
    self * (1 / other)

proc `/`*(other: float, self: Value): Value =
    self * pow(self, -1)

proc tanh*(self: Value): Value = 
    let x = self.data
    let t = (exp(2*x) - 1) / (exp(2*x) + 1)
    let newVal = Value(
        data: t,
        prev: @[self],
    )

    proc diff(): void =
        self.grad += (1 - t*t) * newVal.grad

    newVal.diff = diff
    newVal

proc exp*(self: Value): Value =
    let newVal = Value(
        data: exp(self.data),
        prev: @[self]
    )

    proc diff(): void =
        self.grad += newVal.data * newVal.grad

    newVal.diff = diff
    newVal

proc hash(t: Value): Hash = 
    # Uses the pointer as the hash value
    # Required to put a Value object into a set
    cast[pointer](t).hash


proc backward*(root: Value): void =
    var ordered: seq[Value] = @[]
    var visited = initHashSet[Value]()

    proc buildTopo (v: Value) =
        if v notin visited:
            visited.incl(v)
            for child in v.prev:
                buildTopo(child)
            ordered.add(v)

    buildTopo(root)

    root.grad = 1
    for node in ordered.reversed():
        if node.label != "": echo node
        node.diff()

proc sum*(values: openArray[Value]): Value = 
    var acc = values[0]
    for v in values[1 .. ^1]:
        acc = acc + v
    acc

# let x1 = Value(data: 2, label: "x1")
# let w1 = Value(data: -3, label: "w1")

# let x2 = Value(data: 0, label: "x2")
# let w2 = Value(data: 1, label: "w2")

# let b = Value(data: 6.8813735870195432, label: "b")

# let x1w1 = x1 * w1
# x1w1.label = "x1w1"
# let x2w2 = x2 * w2
# x2w2.label = "x2w2"

# let x1w1_x2w2 = x1w1 + x2w2
# x1w1_x2w2.label = "x1w1_x2w2"
# let x1w1_x2w2_b = x1w1_x2w2 + b
# x1w1_x2w2_b.label = "x1w1_x2w2_b"

# let y = tanh(x1w1_x2w2_b)
# y.label = "y"
# y.grad = 1
# y.backward()

# let a = Value(data: -2, label: "a")
# let b = Value(data: 3, label: "b")

# let d = a * b; d.label = "d"
# let e = a + b; e.label = "e"
# let f = d * e; f.label = "f"
# f.backward()

# let x = Value(data: 3, label: "x")
# let z = Value(data: 7, label: "z")

# let a = pow(x, 5); a.label = "a"
# let b = a - z; b.label = "b"
# let c = exp(x); c.label = "c"
# let d = c * z; d.label = "d"
# let e = b / 2; e.label = "e"
# let y = e + d; y.label = "y"


