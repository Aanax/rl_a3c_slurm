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


----------
ветка 2level_try1
пробую сделать 2 уровня из 2378 noRMS batch 32



-------
Solutions

Дельты через адвантадж (альтернатива - одношаговые ТД) [DONE] delta = advantage.detach(), используется вместо gae (убрано накопление gae*gamma*tau)

Rmsnorm after encoder1?

use crossentropy explicitly instead of KL - sum(p ln q) [DONE] ce_i = -(target_prob * pred_log_prob).sum(dim=1).mean()

make last run not change options!

per option bootstrap for interactor [DONE] interactor_running_target resets at real option-change boundaries (compared via player.actions2[i] vs actions2[i-1], not the raw termination-attempt flag)