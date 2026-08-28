import torch
for p in ['save/model_bioanchor_gsva_decorr_M4/checkpoint_200.tar',
          'save/model_ucec_gsva_decorr_M4/checkpoint_200.tar', 
          'save/model_bioanchor_gsva_decorr_M4_mlp/checkpoint_150.tar',]:
    try:
        c = torch.load(p, map_location='cpu', weights_only=False)
        print(p.split('/')[1], '-> bio_dim:', c.get('bio_dim'),
              '| n_anchors:', c.get('n_anchors'),
              '| head:', c.get('head_type'), '| k:', c.get('n_clusters'))
    except FileNotFoundError:
        print(p, 'NOT FOUND — run: ls save/ | grep -i m4')