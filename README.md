# Behavioral cloning
Teaching Self driving car how to steer in a simulator

## Deriving and designing model

An abstraction of pipeline is created in the model.py
This allowed me to quickly switch between different architectures.

Initially, I tried Nvidia's model:

https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

with my own recorded data from the keyboard simulator, but it did not produce 
smooth angles, which led to many 0 zero data points, which in turn led to
the neural network getting stuck in a local optima and predicing a constant angle.

I tried desperately to avoid this stucking using Dropout and L2 regularization, but 
of course this did not help, because the problem was in the data.

At this point, I also tried VGG16, Comma.ai's model, and my own model based on Comma.ai,
but none of them really worked.