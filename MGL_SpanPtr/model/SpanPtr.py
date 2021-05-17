import torch
import torch.nn as nn
from torch.autograd import Variable

class MGDST(nn.Module):
    def __init__(self, lang, hidden_size, dropout, n_op, slot_meta):
        super(MGDST, self).__init__()
        self.hidden_size = hidden_size
        self.lang = lang
        self.encoder = Encoder(self.lang.n_words, hidden_size, dropout)
        self.decoder = Decoder(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, dropout, n_op, slot_meta)

    def forward(self, input_ids, input_lens, max_value, op_ids=None, teacher=None):

        enc_outputs = self.encoder(input_ids=input_ids.transpose(0, 1),
                                   input_lens=input_lens.cpu())
        
        encoder_outputs, text_hidden = enc_outputs
        state_scores, gen_scores = self.decoder(input_ids, encoder_outputs, text_hidden, max_value, teacher)

        return state_scores, gen_scores

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru_text = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=True)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 0.1)
        self.dropout = nn.Dropout(dropout)

    def get_state(self, bsz, x):
        """Get cell states and hidden states."""
        return Variable(torch.zeros(2, bsz, self.hidden_size)).to(x.device)

    def forward(self, input_ids, input_lens):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        hidden = self.get_state(input_ids.size(1), input_ids)
        if input_lens is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=False)
        outputs, hidden = self.gru_text(embedded, hidden)
        if input_lens is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        encoder_outputs = outputs.transpose(0, 1)
        text_hidden = hidden.unsqueeze(0)
        return encoder_outputs, text_hidden

class Decoder(nn.Module):
    def __init__(self, lang, embedding, vocab_size, hidden_size, dropout, n_op, slot_meta):
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.lang = lang
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.action_cls = nn.Linear(hidden_size, n_op)
        self.n_op = n_op
        self.slots = slot_meta

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot not in self.slot_w2i.keys():
                self.slot_w2i[slot] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, x, encoder_output, encoder_hidden, max_len, teacher=None):
        mask = x.eq(self.pad_idx)

        # Get the slot embedding
        slot_emb_dict = {}
        for slot in self.slots:
            # Slot embbeding
            slot_w2idx = [self.slot_w2i[slot]]
            slot_w2idx = torch.tensor(slot_w2idx)
            slot_w2idx = slot_w2idx.to(x.device)
            slot_emb = self.Slot_emb(slot_w2idx)
            slot_emb_dict[slot] = slot_emb

        batch_size = x.size(0)
        all_point_outputs = torch.zeros(len(self.slots), batch_size, max_len, x.size(1)).to(x.device)
        all_gate_outputs = torch.zeros(len(self.slots), batch_size, self.n_op).to(x.device)
        result_dict = {}
        for j, slot in enumerate(self.slots):
            #w = state_in[:, j].unsqueeze(1)  # B,1,D
            slot_emb = slot_emb_dict[slot]
            w = slot_emb.expand(batch_size, 1, self.hidden_size)
            hidden = encoder_hidden
            slot_value = []
            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D
                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)
                attn_history = nn.functional.softmax(attn_e, -1)  # B,T
                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
                if k == 0:
                    all_gate_outputs[j] = self.action_cls(context.squeeze(1))
                
                p_final = attn_history + 1e-12
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx])

                if teacher is not None:
                    index = teacher[:, j, k].view(batch_size, 1)
                    decoder_input = torch.gather(x, dim=1, index=index)
                    decoder_input = decoder_input.squeeze(1)
                else:
                    index = w_idx.view(batch_size, 1)
                    decoder_input = torch.gather(x, dim=1, index=index)
                    decoder_input = decoder_input.squeeze(1)
                w = self.embedding(decoder_input).unsqueeze(1)

                all_point_outputs[j, :, k, :] = p_final

        return all_gate_outputs.transpose(0, 1), all_point_outputs.transpose(0, 1)
