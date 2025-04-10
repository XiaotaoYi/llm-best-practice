# 中文医疗问答数据集

## 数据

### 数据中有6个文件夹分别是:
>**<Andriatria_男科>**  94596个问答对
> **<IM_内科>**        220606个问答对
**<OAGD_妇产科>**      183751个问答对
**<Oncology_肿瘤科>**   75553个问答对
 **<Pediatric_儿科>**  101602个问答对
 **<Surgical_外科>**   115991个问答对
 总计 792099条数据

### 每个文件夹下有一个csv文件，其中的数据格式为:

department | title | question |  answer
:-: | :-: | :-: | :-: 
心血管科|高血压患者能吃党参吗？|我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗？|高血压病人可以口服党参的。党参有降血脂，降血压的作用，可以彻底消除血液中的垃圾，从而对冠心病以及心血管疾病的患者都有一定的稳定预防工作作用，因此平时口服党参能远离三高的危害。另外党参除了益气养血，降低中枢神经作用，调整消化系统功能，健脾补肺的功能。感谢您的进行咨询，期望我的解释对你有所帮助。|
消化科|哪家医院能治胃反流|烧心，打隔，咳嗽低烧，以有4年多| 建议你用奥美拉唑同时，加用吗丁啉或莫沙必利或援生力维，另外还可以加用达喜片|

## Qwen2.5-7B 微调

### 数据格式

```json
    {
        "instruction": "你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答",
        "input": "2018年5月做了双侧卵巢囊肿摘除术，由于也是子宫内膜异位，所以术后打了六针亮丙瑞林微球，10月11日打完最后一针，之后11月，2019年1月，2月都吃过一次紧急避孕药，打完亮丙瑞林最后一针后已有4个半月还没来月经了，想问问何时才会来月经?",
        "label": "根据你所描述的情况来看的话，你这种情况可能还要专心仔细观察。你吃过紧急避孕药也没再次出现撤退性出血，而且你46岁这个年龄，很容易踏入更年期。对于卵巢囊肿严重患者来说，建议尽早就诊，根据医生的意见来尽早救治，不要盲目的动用药物，以免加重病情。",
        "output[微调前]": "在您描述的情况下，使用亮丙瑞林（也称为醋酸曲普瑞林）进行治疗是为了抑制排卵和调节激素水平，以达到控制子宫内膜异位症的症状。这种药物可以导致一段时间内的闭经，通常持续时间因人而异，可能从几个月到一年不等。停止注射后的恢复过程取决于多种因素，包括个人的身体状况、药物的剂量、以及停药前身体对药物反应的情况。
        一般来说，在停止使用亮丙瑞林后，身体需要一定的时间来调整激素水平，恢复正常周期。有些人可能会在几周内开始出现月经迹象，而另一些人则可能需要数月的时间才能恢复正常的月经周期。在您的情况下，已经过去大约四个月没有月经，并且在这段时间内还使用了紧急避孕药，这可能进一步影响了月经的恢复。紧急避孕药中的激素成分可能会干扰正常周期，有时会导致月经延迟或提前。因此，紧急避孕药可能是导致您当前月经延迟的原因之一。建议您咨询妇科医生或内分泌科专家，进行详细的检查和评估。医生可能会建议您进行血液激素测试、超声波检查或其他相关检测，以了解体内激素水平是否已恢复正常，并确定是否需要进一步的治疗",
        "output[微调后]": "这种情况有可能是内分泌失调引来的闭经现象，建议你到医院做个性激素六项的仔细检查，在医生指导下服食药物调养。平时要多留意歇息，防止操劳和精神紧张，始终保持心情舒畅，多吃许多营养丰富易补充的食物，少吃辛辣打击极冷的食物，确保充足的深度睡眠时间。"
    },
    {
        "instruction": "你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答",
        "input": "我自己知道有这个毛病，已经很久了，有时候会很难受别扭，但是过一会就没事了，但是我想治疗了，但是不知道咋办呢，请问胎粪会使肠管扩张吗",
        "label": "你好，不好确定，胎儿肠道扩张的原因有很多。比如受先天因素的影响，即是胎儿本身先天肠道畸型；有的是因为胎粪堵塞引起等等。后者的肠管扩张是可以自愈的。通过彩超不能够确诊肠管扩张就是是肠道畸形。肠道畸形和肠管扩张还是有区别的。",
        "output[微调前]": "胎粪的排出确实可能导致一定程度上的肠管扩张，特别是对于新生儿来说。在正常情况下，胎儿在子宫内就已经开始形成胎粪（即胎儿消化过程中产生的废物）。当胎儿出生后，医生通常会在第一小时内帮助清除胎粪，以避免肠道堵塞和相关并发症的发生。如果胎粪排除不畅或者出现其他阻塞情况，确实可能引起轻微的肠管扩张症状。肠管扩张本身是一种较为常见的临床表现，在新生儿中尤其常见，原因包括但不限于：胎粪堵塞、腹胀、消化不良、感染等。如果您或您的孩子存在持续的不适感，并怀疑与胎粪有关的肠管扩张，建议及时就医进行详细检查和诊断。治疗方法通常会根据具体病因而定：
        1. **轻度肠管扩张**：通过观察和调整喂养方式来缓解症状。
        2. **胎粪堵塞**：可能需要采用通便措施，如使用灌肠、按摩或其他医学手段帮助排出胎粪。
        3. **感染**：如果是由于感染引起的肠管扩张，则需按照医嘱使用抗生素和其他抗炎药物。
        4. **严重情况**：在某些极端情况下，可能需要紧急手术干预。

        请务必咨询儿科医生或相应专科医生",
        "output[微调后]": "考虑是消化不良引起的，建议可以给孩子用鸡内金散、小儿健脾散、整肠生、施尔康等进行调理看看，如果效果不佳时，可以到儿科进行系统的检查确诊。，你好，你描述的考虑是乳腺炎引起乳房肿块疼痛，一般是由于细菌感染所致，常见于哺乳期妇女，可以考虑使用青霉素或头孢菌素类药物抗炎治疗"
    },
```

### 结果
在Qwen2.5-7B上微调的结果，随机选择了1/200的数据进行微调，下表为评估结果。

|   分数  |  Qwen2.5-7B | LoRA (r=16) |
| ------- | -------- | ----- |
| BLEU-4  |  0.0100    | 0.0393  |
| Rouge-1 |  0.0371   | 0.0426 |
| Rouge-2 |  0.0099    | 0.0017  |
| Rouge-l |  0.0335   | 0.0426 |





            

