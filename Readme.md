# 合同阅读答题

通过阅读理解的方式进行合同内相关实体的抽取。
## example
**段落:**  
"上海银行资产池业务-现>912102QR-0269上海子元汽车零部件有限公司2020供货商:吉林省通用机械(集团)有限责任公司年供应商供货价格)J编号:供货商编号:04.031制表日期:2020-04-27部门:采购中心资产池业交货地点:嘉安公路3555号序号系统代码零件号零件名称使用车型单位下降比率本年度单价备注11.02.06.00453CC 805 615拖钩B-SUV工具盒件8.940021.02.06.00415NG 805 615拖钩TiguanNF工具盒件8.0000务-现金说明:1. 价格为不含税价,且只供供货商准备原材料用。2.供货商应按双方签订的供货协议及相关附件中各项条款的要求供货,所交货物由我公司验收合格后办理入库手续。上3.货物的包装必须符合便于保存、堆放、计量等基本要求,每批货物的送货单据必须注明货物的系统代码5.有效期自2020年01月01日至2020年12月31日止。4.为保证生产,供货商应储备相应的库存(储备资金自负),以备我公司追加要货。6.本价格单执行双方签订的《国产零部件和生产材料采购条款》之条文。渗银行资产汇上海子元汽车零部件有限公司吉林省通用机械(集团)有限责住公司采购方代表:授权签章:务-现金供货商代表供货商签章:每银行资产池业号-现金期:银行资产池业务-现金日期:资产池业务-现金上海银行资产池业"

**问题:**  
合同买房是谁？

**抽取结果:**  
上海子元汽车零部件有限公司

## data
进行过实体标注的合同语料。
## models
暂时提供了两种可使用的模型：
* BERT ------"luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"
* 经过P-tuning-v2编辑过的BERT。作为一种prompt learning的范式，该模型在小样本下有更强的表现。
## Requirements
```
pip install -r /path/to/requirements.txt
```

## training
```
python3 train.py
```