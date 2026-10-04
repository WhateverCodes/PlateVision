from pathlib import Path
import yaml,json
root=Path.cwd();folder=root/'outputs/PlateVision1/data/ocr_finetune_audit_v1'
c=yaml.safe_load((root/'work/en_ocr_train.yaml').read_text())
c['Global'].update(use_gpu=False,distributed=False,epoch_num=1,print_batch_step=1,save_model_dir=str(root/'work/ocr_finetune_trial'),pretrained_model=str(root/'work/paddle_downloads/en_PP-OCRv5_mobile_rec_pretrained'),eval_batch_step=[0,20])
c['Optimizer']['lr'].update(learning_rate=.00001,warmup_epoch=0)
c['Train']['dataset']['data_dir']=str(folder);c['Train']['dataset']['label_file_list']=[str(folder/'train.txt')]
c['Train']['dataset']['transforms']=[v for v in c['Train']['dataset']['transforms'] if 'RecConAug' not in v]
c['Train']['sampler'].update(first_bs=2,fix_bs=True,scales=[[320,48]])
c['Train']['loader'].update(batch_size_per_card=2,num_workers=0,drop_last=False)
c['Eval']['loader'].update(batch_size_per_card=2,num_workers=0,shuffle=False)
# Evaluation manifest still needs a separately audited text-line view;
# do not substitute training examples as validation.
c['Eval']['dataset']['data_dir']=str(folder)
c['Eval']['dataset']['label_file_list']=[str(folder/'validation_PENDING_AUDIT.txt')]
(root/'work/ocr_cpu_trial_DRAFT.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
print('Draft CPU config saved; intentionally cannot run until validation audit is ready.')
