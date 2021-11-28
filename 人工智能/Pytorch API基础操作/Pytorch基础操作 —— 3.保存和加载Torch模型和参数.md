
@[toc]

# 基础操作 —— 模型的存储

我们的网络训练完成后，如果表现还不错，通常会想到把模型保存起来。所谓的模型，指的是我们用什么的神经层构建的网络，而与模型一同进行保存的，还有与模型相关的参数解。

所以从这个概念出发，我们可以有两种保存和读取方法。

## S/L 模型

这种会把模型和相关参数一起保存起来，这种做法能很方便我们在做其他工程时，把之前的网络集成到新的应用里。

```python
# save the model 
torch.save(the_model, PATH)

# load the model
the_model = torch.load(PATH)
```

但是它会有个缺点，就是保存的文件体积特别大，因为它不光保存了模型，也保留了相关参数。所以有时候我们会想，如果我们已知了某种网络结构，是否可以只保留参数？

## S/L 参数
这便是上面提到的第二种方法，就是仅存储模型的训练参数，这种方法有个前置条件，就是要求使用者已知网络模型。

```python
# save the arguments
torch.save(the_model.state_dict(), PATH)

# load the arguments
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

在某些时候，我们也可以使用这个方法来调整网络的预训练模型，比如在做YOLO的时候，也经常会用到这种类似的方法，但这一部分我就不在这篇文章里展开了。

# TorchScript 

以上两种方法，适用于Python对Python的项目，如果是Python对C/CPP，Python对Java的项目，就不能使用上面提到的两种方法。

这时我们会使用第三种，把模型以TorchScript的形式进行保存，并加载。比方说我们训练了一个网络模型，它能识别文字信息，现在我们需要把相关任务以TorchScript的形式进行保存，这样就可以集成到比如C/C++程序，或者Java程序里。

## Pytorch模型转成TorchScript
~~~python
	# define a neural network module
	model = DefinedNeuralNetworkModule

    # converting to Torch Script via Annotation
    serialized_model = torch.jit.script(model)

    # save the torch script for C++
    serialized_model.save("LSTM_Classfication.pt")
~~~

## C/C++加载TorchScript

~~~cpp
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model." << std::endl;
        return -1;
    }
~~~
