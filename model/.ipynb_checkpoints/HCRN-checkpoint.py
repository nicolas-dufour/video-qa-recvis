import numpy as np
from torch.nn import functional as F
from transformers import AutoModel

from .utils import *
from .CRN import CRN, CRNCondAblation

###### BASE HCRN CODE #######

class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill


class InputUnitLinguisticGlove(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguisticGlove, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)
        
        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding


class InputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisual, self).__init__()

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(appearance_video_feat.size(1)):
            clip_level_motion = motion_video_feat[:, i, :]  # (bz, 2048)
            clip_level_motion_proj = self.clip_level_motion_proj(clip_level_motion)

            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj, dim=1),
                                                                clip_level_motion_proj)
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitCount, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.regression = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.regression(out)

        return out


class HCRNNetworkGlove(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type):
        super(HCRNNetworkGlove, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguisticGlove(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguisticGlove(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguisticGlove(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes,module_dim=module_dim)

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,
                question_len):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question, question_len)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            out = self.output_unit(question_embedding, visual_embedding)
        else:
            question_embedding = self.linguistic_input_unit(question, question_len)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out
    
    


###### ORIGINAL CODE #######

class InputUnitLinguisticTransformer(nn.Module):
    def __init__(self,module_dim=512, transformer_path = 'bert-base-uncased',transformer_cache_dir =None, train_bert = 'freeze', mult_embedding=False):
        super(InputUnitLinguisticTransformer, self).__init__()

        self.module_dim = module_dim
        self.train_bert = train_bert
        self.mult_embedding = mult_embedding
        self.transformer_path = transformer_path

        self.bert = AutoModel.from_pretrained(transformer_path,output_hidden_states=True,cache_dir=transformer_cache_dir)
        self.bert_dim = 768 #self.bert.encoder.layer[-1].output.dense.out_features
#         self.bert.pooler = nn.Identity()
        
        if(self.train_bert == 'freeze'):
            print('Freeze all layers')
            for param in self.bert.parameters():
                param.requires_grad = False
        elif(self.train_bert == 'last-2'):
            print('Freeze all except last 2 layers')
            for name, param in self.bert.named_parameters():
                layer_number = name.split('.')[2]
                if(not (layer_number in ['10','11','pooler'])):
                    param.requires_grad = False
        elif(self.train_bert == 'last-4'):
            print('Freeze all except last 4 layers')
            for name, param in self.bert.named_parameters():
                layer_number = name.split('.')[2]
                if(not (layer_number in ['8','9','10','11','pooler'])):
                    param.requires_grad = False
            
            
        
        if(mult_embedding):
            self.embedding = nn.Linear(4*self.bert_dim,self.module_dim)
        else:
            self.embedding = nn.Linear(self.bert_dim,self.module_dim)
            
        self.embedding_dropout = nn.Dropout(0.15)
        self.embedding_elu = nn.ELU()


    def forward(self, questions_input_ids, question_attention_mask, question_token_type_ids):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        if(self.transformer_path == 'bert-base-uncased'):
            data = {'input_ids':questions_input_ids,'attention_mask':question_attention_mask,'token_type_ids':question_token_type_ids}
        else:
            data = {'input_ids':questions_input_ids,'attention_mask':question_attention_mask}
        bert_output = self.bert(**data)
        if(self.train_bert == 'freeze'):
            hidden_states = bert_output[2]
            if(not self.mult_embedding):
                question_embedding = torch.mean(hidden_states[-1], dim=1)
            else:
                last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
                cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
                question_embedding = torch.mean(cat_hidden_states, dim=1)
        else:
            question_embedding = torch.mean(bert_output[0], dim=1)
        question_embedding = self.embedding(question_embedding)
        question_embedding = self.embedding_elu(self.embedding_dropout(question_embedding))
        
        return question_embedding

class HCRNNetworkBert(nn.Module):
    def __init__(self, vision_dim, module_dim, k_max_frame_level, k_max_clip_level, spl_resolution, question_type, vocab=None, transformer_cache_dir=None, transformer_path = 'bert-base-uncased', train_bert = False, mult_embedding=False):
        super(HCRNNetworkBert, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        if self.question_type in ['action', 'transition', 'tvqa']:
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path, transformer_cache_dir=transformer_cache_dir, train_bert = train_bert, mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path,transformer_cache_dir=transformer_cache_dir, train_bert = train_bert, mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path,transformer_cache_dir=transformer_cache_dir, train_bert = train_bert, mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes,module_dim=module_dim)

        init_modules(self.modules(), w_init="xavier_uniform")

    def forward(self, ans_candidates_tokens, ans_candidates_attention_mask, ans_candidates_token_type_ids, video_appearance_feat, video_motion_feat,
                question_tokens,question_attention_masks,question_token_type_ids):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question_tokens.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question_tokens,question_attention_masks,question_token_type_ids)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            out = self.output_unit(question_embedding, visual_embedding)
        else:
            question_embedding = self.linguistic_input_unit(question_tokens,question_attention_masks,question_token_type_ids)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_tokens, ans_candidates_attention_mask, ans_candidates_token_type_ids)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out

class InputUnitVisualAblation(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512,ablated_features=[]):
        super(InputUnitVisualAblation, self).__init__()
        if("clip_motion" in ablated_features):
            self.clip_level_motion_cond = CRNCondAblation(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        else:
            self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        if("clip_question" in ablated_features):
            self.clip_level_question_cond = CRNCondAblation(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        else:
            self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        if("video_motion" in ablated_features):
            self.video_level_motion_cond = CRNCondAblation(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        else:
            self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        if("video_question" in ablated_features):
            self.video_level_question_cond = CRNCondAblation(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)
        else:
            self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)
        
        self.appearance_video_feat_ablation = "video_appearance_ablation" in ablated_features
        
        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        if(self.appearance_video_feat_ablation):
            appearance_video_feat = torch.zeros(appearance_video_feat.size(),device = appearance_video_feat.device)
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(appearance_video_feat.size(1)):
            clip_level_motion = motion_video_feat[:, i, :]  # (bz, 2048)
            clip_level_motion_proj = self.clip_level_motion_proj(clip_level_motion)

            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj, dim=1),
                                                                clip_level_motion_proj)
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output
    
class HCRNNetworkBertAblation(nn.Module):
    def __init__(self, vision_dim, module_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type, transformer_cache_dir=None, transformer_path = 'bert-base-uncased', train_bert = False, mult_embedding=False, ablated_features =[]):
        super(HCRNNetworkBertAblation, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        if self.question_type in ['action', 'transition']:
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path,transformer_cache_dir=transformer_cache_dir, train_bert = train_bert, mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisualAblation(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim, ablated_features=ablated_features)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path,transformer_cache_dir=transformer_cache_dir, train_bert = train_bert, mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisualAblation(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim, ablated_features=ablated_features)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path,transformer_cache_dir=transformer_cache_dir, train_bert = train_bert, mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisualAblation(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim,ablated_features=ablated_features)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes,module_dim=module_dim)

        init_modules(self.modules(), w_init="xavier_uniform")

    def forward(self, ans_candidates_tokens, ans_candidates_attention_mask, ans_candidates_token_type_ids, video_appearance_feat, video_motion_feat,
                question_tokens,question_attention_masks,question_token_type_ids):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question_tokens.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question_tokens,question_attention_masks,question_token_type_ids)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            out = self.output_unit(question_embedding, visual_embedding)
        else:
            question_embedding = self.linguistic_input_unit(question_tokens,question_attention_masks,question_token_type_ids)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_tokens, ans_candidates_attention_mask, ans_candidates_token_type_ids)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out
    
class SubtitlesSelection(nn.Module):
    def __init__(self, module_dim=512):
        super(SubtitlesSelection, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.s_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, sub_feat):
        sub_feat = self.dropout(sub_feat)
        q_proj = self.q_proj(question_rep)
        s_proj = self.s_proj(sub_feat)
        s_q_cat = torch.cat((s_proj, q_proj * s_proj), dim=-1)
        s_q_cat = self.cat(s_q_cat)
        s_q_cat = self.activation(s_q_cat)

        return s_q_cat
    
class InputUnitVisualSubtitles(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisualSubtitles, self).__init__()

        self.clip_level_sub_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_sub_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        self.sequence_encoder = nn.LSTM(module_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_sub_proj = nn.Linear(module_dim, module_dim)
        self.sub_q_clip_conditioning = SubtitlesSelection(module_dim)
        self.video_level_sub_proj= nn.Linear(module_dim, module_dim)
        self.sub_q_video_conditioning = SubtitlesSelection(module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)
        
        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, subtitle_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(appearance_video_feat.size(1)):
            clip_level_subtitle = subtitle_video_feat[:, i, :]  # (bz, 2048)
            clip_level_sub_proj = self.clip_level_sub_proj(clip_level_subtitle)
            clip_level_sub_proj = self.sub_q_clip_conditioning(question_embedding_proj,clip_level_sub_proj)
            
            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance) # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_sub = self.clip_level_sub_cond(torch.unbind(clip_level_appearance_proj, dim=1),
                                                                clip_level_sub_proj)
            
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_sub, question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level subs
        _, (video_level_subtitle, _) = self.sequence_encoder(subtitle_video_feat.float())
        video_level_subtitle = video_level_subtitle.transpose(0, 1)
        video_level_subtitle_feat_proj = self.video_level_sub_proj(video_level_subtitle)
        # video level CRNs 
        
        video_level_subtitle_feat_proj = self.sub_q_video_conditioning(question_embedding_proj,video_level_subtitle_feat_proj[:,0,:]).unsqueeze(1)
        video_level_crn_sub = self.video_level_sub_cond(clip_level_crn_outputs, video_level_subtitle_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_sub,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)
        

        return video_level_crn_output

class HCRNNetworkTVQA(nn.Module):
    def __init__(self, vision_dim, module_dim, k_max_frame_level, k_max_clip_level, spl_resolution, question_type, vocab=None, transformer_cache_dir=None, transformer_path = 'bert-base-uncased', train_bert = False, mult_embedding=False):
        super(HCRNNetworkTVQA, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        if self.question_type in ['action', 'transition', 'tvqa']:
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path, transformer_cache_dir=transformer_cache_dir, train_bert = train_bert,
                                                                        mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisualSubtitles(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim,
                                                              module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path,transformer_cache_dir=transformer_cache_dir, train_bert = train_bert,
                                                                        mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisualSubtitles(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim,
                                                              module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path,transformer_cache_dir=transformer_cache_dir, train_bert = train_bert,
                                                                        mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisualSubtitles(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim,
                                                              module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes,module_dim=module_dim)
            
    
        
        init_modules(self.modules(), w_init="xavier_uniform")

    def forward(self, ans_candidates_tokens, ans_candidates_attention_mask, ans_candidates_token_type_ids, video_appearance_feat,
                question_tokens,question_attention_masks,question_token_type_ids,
               subtitles_tokens,subtitles_attention_mask, subtitles_token_type_ids):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question_tokens.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question_tokens,question_attention_masks,question_token_type_ids)
            video_subtitle_feat = []
            for i in range(subtitles_tokens.size(1)):
                video_subtitle_feat.append(self.linguistic_input_unit(subtitles_tokens[:,i,:],subtitles_attention_mask[:,i,:], subtitles_token_type_ids[:,i,:]))
            video_subtitle_feat = torch.stack(video_subtitle_feat,dim=1)
            
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_subtitle_feat, question_embedding)

            visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            out = self.output_unit(question_embedding, visual_embedding)
        else:
            question_embedding = self.linguistic_input_unit(question_tokens,question_attention_masks,question_token_type_ids)
            video_subtitle_feat = []
            for i in range(subtitles_tokens.size(1)):
                video_subtitle_feat.append(self.linguistic_input_unit(subtitles_tokens[:,i,:],subtitles_attention_mask[:,i,:], subtitles_token_type_ids[:,i,:]))
            video_subtitle_feat = torch.stack(video_subtitle_feat).view(8,batch_size,-1).transpose(1,0)
            
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_subtitle_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])
            
            ans_candidates_embedding = []
            for i in range(ans_candidates_tokens.size(1)):
                ans_candidates_embedding.append(self.linguistic_input_unit(ans_candidates_tokens[:,i,:],ans_candidates_attention_mask[:,i,:], ans_candidates_token_type_ids[:,i,:]))
            ans_candidates_embedding = torch.stack(ans_candidates_embedding).view(5,batch_size,-1).transpose(1,0).reshape(5*batch_size,-1)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
            out = out.view(batch_size,-1)
        return out


class InputUnitVisualStream(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisualStream, self).__init__()
        
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)

        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)
        
        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(appearance_video_feat.size(1)):
            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)

            clip_level_crn_question = self.clip_level_question_cond(torch.unbind(clip_level_appearance_proj, dim=1)
                                                                    , question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)
        # video level CRNs
        video_level_crn_question = self.video_level_question_cond(clip_level_crn_outputs,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output

class InputUnitSubtitlesStream(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, module_dim=512):
        super(InputUnitSubtitlesStream, self).__init__()

        self.video_level_question_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)

        self.sub_proj = nn.Linear(module_dim, module_dim)
        self.question_embedding_proj = nn.Linear(module_dim, module_dim)
        
        self.sub_q_clip_conditioning = SubtitlesSelection(module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, subtitles_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = subtitles_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        subtitles_selected =[]
        for i in range(subtitles_feat.size(1)):
            clip_level_subtitle = subtitles_feat[:, i, :]  # (bz, 2048)
            clip_level_sub_proj = self.sub_proj(clip_level_subtitle)
            clip_level_sub_proj = self.sub_q_clip_conditioning(question_embedding_proj,clip_level_sub_proj)
            subtitles_selected+=[clip_level_sub_proj]

        subtitles_selected = torch.stack(subtitles_selected,dim=1)
        video_level_crn_question = self.video_level_question_cond(torch.unbind(subtitles_selected, dim=1),
                                                                  question_embedding_proj)

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output

class JointFeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(JointFeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.f_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, features):
        features = self.dropout(features)
        q_proj = self.q_proj(question_rep)
        f_proj = self.f_proj(features)

        f_q_cat = torch.cat((f_proj, q_proj.unsqueeze(1) * f_proj), dim=-1)
        f_q_cat = self.cat(f_q_cat)
        f_q_cat = self.activation(f_q_cat)

        attn = self.attn(f_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        f_distill = (attn * features).sum(1)

        return f_distill

class HCRNNetworkTVQA2Stream(nn.Module):
    def __init__(self, vision_dim, module_dim, k_max_frame_level, k_max_clip_level, spl_resolution, question_type, vocab=None, transformer_cache_dir=None, transformer_path = 'bert-base-uncased', train_bert = False, mult_embedding=False):
        super(HCRNNetworkTVQA2Stream, self).__init__()

        self.question_type = question_type
        

        if self.question_type in ['tvqa']:
            self.linguistic_input_unit = InputUnitLinguisticTransformer(transformer_path = transformer_path, transformer_cache_dir=transformer_cache_dir, train_bert = train_bert,
                                                                        mult_embedding = mult_embedding)
            self.visual_input_unit = InputUnitVisualStream(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            
            self.subtitles_input_unit = InputUnitSubtitlesStream(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, module_dim=module_dim)
            
            self.joint_feature_aggregation = JointFeatureAggregation(module_dim)
            self.joint_projection = nn.Linear(2*module_dim,module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        else:
            raise "Only supports TVQA"
            
    
        
        init_modules(self.modules(), w_init="xavier_uniform")

    def forward(self, ans_candidates_tokens, ans_candidates_attention_mask, ans_candidates_token_type_ids, video_appearance_feat,
                question_tokens,question_attention_masks,question_token_type_ids,
               subtitles_tokens,subtitles_attention_mask, subtitles_token_type_ids):
        batch_size = question_tokens.size(0)
        question_embedding = self.linguistic_input_unit(question_tokens,question_attention_masks,question_token_type_ids)
        video_subtitle_feat = []
        for i in range(subtitles_tokens.size(1)):
            video_subtitle_feat.append(self.linguistic_input_unit(subtitles_tokens[:,i,:],subtitles_attention_mask[:,i,:], subtitles_token_type_ids[:,i,:]))
        video_subtitle_feat = torch.stack(video_subtitle_feat).view(8,batch_size,-1).transpose(1,0)
        
        visual_embedding = self.visual_input_unit(video_appearance_feat, question_embedding)
        sub_embedding = self.subtitles_input_unit(video_subtitle_feat,question_embedding)

        q_joint_embedding = self.joint_feature_aggregation(question_embedding, torch.cat([visual_embedding,sub_embedding],dim=1))
                
        

        # ans_candidates: (batch_size, num_choices, max_len)

        batch_agg = np.reshape(
            np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])
            
        ans_candidates_embedding = []
        for i in range(ans_candidates_tokens.size(1)):
            ans_candidates_embedding.append(self.linguistic_input_unit(ans_candidates_tokens[:,i,:],ans_candidates_attention_mask[:,i,:], ans_candidates_token_type_ids[:,i,:]))
        ans_candidates_embedding = torch.stack(ans_candidates_embedding).view(5,batch_size,-1).transpose(1,0).reshape(5*batch_size,-1)

        a_joint_embedding = self.joint_feature_aggregation(ans_candidates_embedding, torch.cat([visual_embedding,sub_embedding],dim=1)[batch_agg])

        out = self.output_unit(question_embedding[batch_agg], q_joint_embedding[batch_agg],
                                ans_candidates_embedding,
                                a_joint_embedding)
        out = out.view(batch_size,-1)
        return out