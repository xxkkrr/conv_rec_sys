import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.manual_seed(0)

seqs = ['gigantic_string','tiny_str','medium_str']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(set(''.join(seqs)))

# make model
embed = nn.Embedding(len(vocab), 10)
lstm = nn.LSTM(10, 5, batch_first=True, bidirectional=True)

vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]

# get the length of each seq in your batch
seq_lengths = torch.LongTensor([len(seq) for seq in vectorized_seqs])

# dump padding everywhere, and place seqs on the left.
# NOTE: you only need a tensor as big as your longest sequence
seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
	seq_tensor[idx, :seqlen] = torch.LongTensor(seq)


# SORT YOUR TENSORS BY LENGTH!
print("seq_tensor:", seq_tensor)
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]
print("seq_tensor:", seq_tensor)

# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors
# seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)

# embed your sequences
seq_tensor = embed(seq_tensor)

# pack them up nicely
packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), True)

# throw them through your LSTM (remember to give batch_first=True here if you packed with it)
packed_output, (ht, ct) = lstm(packed_input)

# unpack your output if required
output, _ = pad_packed_sequence(packed_output, True)
# output = output.transpose(1, 0)
print (output)

# Or if you just want the final hidden state?
print (ht)
ht=ht.view(1, 2, 3, 5)
htht=torch.cat([ht[-1][0],ht[-1][1]], -1)
print(htht)


# REMEMBER: Your outputs are sorted. If you want the original ordering
# back (to compare to some gt labels) unsort them
_, unperm_idx = perm_idx.sort(0)
output = output[unperm_idx]
print (output)
#print (ht[-1][unperm_idx])
