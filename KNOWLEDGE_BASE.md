Сейчас инициализация Conv2d слоев там дефолтная от пайторча версии 1.6.0

Выглядит так 

def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

\\\ Это Хе, только с негатив слоуп =  корнем из 5
Т.е. U(-bound, bound)

И для весов  bound = sqrt(1/in_features) (считал так. из сорсов. там gain=math.sqrt(2.0 / (1 + a ** 2))  и bound = sqrt(3)*gain/sqrt(in_features)) 
https://github.com/pytorch/pytorch/blob/8625ffbd45884464f736cfc61300c14f47633641/torch/nn/init.py#L572

И затем  (вне функции) это все домнажается на relu_gain ~ sqrt(2)

т.е. в итоге sqrt(2)*sqrt(1/in_features)

А биасы зануляются


Для нашего линейного слоя
инициализация чистый хе с релу_гейном

U(-bound,bound)
где bound = sqrt(2) * sqrt(3/fan_in)