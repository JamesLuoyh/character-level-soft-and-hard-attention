# character level soft and hard attention Comparison
Comparison between soft attention and hard attention on character level transduction based on the paper https://arxiv.org/pdf/1808.10024.pdf.
Code is heavily referenced from https://bastings.github.io/annotated_encoder_decoder/

Run command:
python train_phoneme.py --lr 0.0007 --layer 1 --epoch 15 --maxlen 10 --dropout 0.1 --attention hard