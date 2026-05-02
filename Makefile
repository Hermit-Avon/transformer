.PHONY: all
data:
	git clone https://www.modelscope.cn/datasets/SelinaRR/Multi30K.git
	mv Multi30K/Multi30K.zip .
	rm -rf Multi30K
	unzip Multi30K.zip
	mkdir -p data/multi30k
	mv datasets/test/test2016.de data/multi30k/test.de
	mv datasets/test/test2016.en data/multi30k/test.en
	mv datasets/train/train.en data/multi30k/train.en
	mv datasets/train/train.de data/multi30k/train.de
	mv datasets/valid/val.de data/multi30k/valid.de
	mv datasets/valid/val.en data/multi30k/valid.en
	rm -rf datasets
	rm Multi30K.zip

clean:
	rm -rf datasets data
	rm Multi30K.zip
