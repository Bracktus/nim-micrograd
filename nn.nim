import engine
import strformat
import random
import sequtils
import sugar

randomize()

type 
    Neuron = ref object
        weights*: seq[Value]
        bias*: Value

proc newNeuron(size: int): Neuron =
    var weights: seq[Value]
    for i in 0 ..< size:
        weights.add(Value(data: rand(-1.0 .. 1.0)))
    let bias = Value(data: rand(-1.0 .. 1.0))
    Neuron(weights: weights, bias: bias)

proc value(self: Neuron, x: seq[Value | float]): Value =
    result = sum(zip(self.weights, x).map(v => v[0] * v[1])) + self.bias
    result = tanh(result)

proc parameters(self: Neuron): seq[Value] =
    self.weights & @[self.bias]

proc `$`(self: Neuron): string =
    fmt"Neuron(weights: {self.weights}, bias: {self.bias})"

type
    Layer = ref object
        neurons*: seq[Neuron]

proc newLayer(sizeIn, sizeOut: int): Layer =

    var neurons: seq[Neuron] = @[]
    for i in 0 ..< sizeOut:
        neurons.add(newNeuron(size = sizeIn))
    Layer(neurons: neurons)

proc value(self: Layer, x: seq[Value | float]): seq[Value] =
    for i, neuron in self.neurons:
        result.add(neuron.value(x))

proc parameters(self: Layer): seq[Value] =
    for neuron in self.neurons:
        result = result.concat(neuron.parameters())

proc `$`(self: Layer): string =
    fmt"Layer(neurons: {self.neurons})"

type
    MLP = ref object
        layers*: seq[Layer]

proc newMLP*(sizeIn: int, sizeOuts: openArray[int]): MLP =
    let layerSizes = @[sizeIn] & @sizeOuts
    var layers: seq[Layer] = @[]
    for i in 0 ..< layerSizes.len - 1:
        let layer = newLayer(layerSizes[i], layerSizes[i + 1])
        layers.add(layer)

    MLP(layers: layers)

proc value(self: MLP, x: openArray[Value]): seq[Value] =
    result = @x
    # This passes the result of one layer to the next one
    for layer in self.layers:
        result = layer.value(result)

proc value(self: MLP, x: openArray[float]): seq[Value] =
    self.value(x.map(v => Value(data: v)))

proc parameters(self: MLP): seq[Value] =
    for layer in self.layers:
        result = result.concat(layer.parameters())

proc loss(self: MLP, xs: seq[seq[float]], ys: openArray[float]): Value =
    var lossValues: seq[Value]
    let ypred = xs.map(v => self.value(v)[0])
    for (ygt, yout) in zip(ys, ypred):
        let err = ygt - yout
        lossValues.add(err * err)

    sum(lossValues)

proc `$`(self: MLP): string =
    fmt"MLP(layers: {self.layers})"

let n = newMLP(3, [4, 4, 1])
let xs = @[
    @[2.0, 3.0, -1.0],
    @[3.0, -1.0, 0.5],
    @[0.5, 1.0, 1.0],
    @[1.0, 1.0, -1.0]
]
let ys = [1.0, -1.0, -1.0, 1.0]

for i in 0 ..< 100:
    var l = n.loss(xs, ys)

    for p in n.parameters:
        p.grad = 0
    l.backward()

    for p in n.parameters:
        p.data += 0.075 * -p.grad

    echo fmt"step: {i}, loss: {l.data}"

echo n


