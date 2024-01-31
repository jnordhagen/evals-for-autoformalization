

def train_to_af_for_maf(model,
                        formal_data_set, # e.g., ITP lib like mathlib
                        informal_data_set,  # e.g., time-tested maths textbook e.g., Rudin, CLRS.
                        ):
    for (nl, fl_gt) in formal_data_set:
        for (nl_gt, fl) in informal_data_set:
    
            # -- Learn to Formalize: nl_i->fl_gt from fl_gt -> nl_i_i -> fl_gt
            nl_i_i = model("informalize " + fl_gt, sampling=top_p, num_out=k)  # noise is good for robustness!
            fl_i_i = model("formalize " + nl_i_i, sampling=random, num_out=1)  # num_out = 1 => learns equivalences! todo: num_out=1 good?
    
            # - Train Step to Formalize from high quality formal dataset ~ opt.step((nl_i -> fl_gt)_i) 
            loss = loss_fn([fl_gt, fl_i]_i); loss.backward() 
            loss = lss_fn([nl, nl_i]_i); loss.backward() if nl != '' else None
            # if data improves val loss e.g., on Proofnet, save for future use!
        
            # -- Learn to Informalize: fl_j->nl_gt from nl_gt -> fl_j_j -> nl_gt
            fl_j_j := model('formalize ' + nl_gt, sampling=top_p, num_out=k)
            nl_j_j := model('informalize ' + fl_j_j, sampling=random, num_out=1)
            # - Train Step to Informalize from high quality informal dataset ~ opt.step([fl_j -> nl_gt])
            loss = loss_fn([nl_j, nl_gt]); loss.backward()
            loss = lss_fn([fl, fl_j]_j); loss.backward() if fl != '' else None
        
            # -- Predict Proofs given Theorem (proofs are hard to come up with, so whenever available learn to generate them)
            # - IT -> IPf (half round trip training) todo: careful with predicting only sledgehammer
            ipf_i = model('generate informal proof ', it)  # model can infer it's from an informal theorem given the theorem will be in nl so I think this shorter prompt is better, we are fine-tuning anyway.
            loss = loss_fn([ipf_i]_i, ipf*); loss.backward()
            it_i = model('generate informal theorem ', it)
            loss = loss_fn([it_i]_i, it*); loss.backward()  # just autoregressive loss, so it knows the theorem. I've learned that ICL isn't actually as good as fine-tuning based on my daily use of GPT4/Claude 2.0. 
        
            # - FT -> FPf (half round trip training) todo: careful with predicting only sledgehammer
            fpf_i = model('generate formal proof ', ft)  # model can infer it's from an form theorem given the theorem will be in fl, shorter prompt better likely + we are already fine-tuning. 
            loss = loss_fn([fpf_i]_i, fpf*); loss.backward()
            ft_i = model('generate formal theorem ', ft)
            loss = loss_fn([ft_i]_i, ft*); loss.backward()  # just autoregressive loss, so it knows the theorem. I've learned that ICL isn't actually as good as fine-tuning based on my daily use of GPT4/Claude 2.0. 
        
            # -- Standard Autoregressive training: Predict raw formal and informal texbook data (pre-pre-training)
            loss = loss_fn(model(fl_gt), fl_gt); loss.backward()
            loss = loss_fn(model(nl_gt), nl_gt); loss.backward()
            # loss = loss_fn(model(fl), fl); loss.backward()
            # loss = loss_fn(model(nl), nl); loss.backward()

            # -- Jointly train everything (for better hardware usgae)
            opt.step() # trains all tasks: nl->fl, fl->nl, it->ipf, ipf->it, ft->fpf, fpf->ft 
            opt.zero_grad()  # zero grads of only params the opt is optimizing
        
            # -- Stop when target its/num_tokens met
            stop(it == target_its)
    return model # for clarify of code, but it likely mutates it :( oh no side-effect...


if __name__ == '__main__':
    # Train with AF4MAF Back Translation based
    model = train_to_af_for_maf(model, formal_data_set, informal_data_set)

    # Eval if model improved from train procedure for MAF
    print('---- Display if our AF4MAF training improved eval metrics on benchmarks ----')
    eval_af(model, eval_data_set=af_dataset, metrics=(ppl, exact_str_match, avg_acc, token_edit_distance))
    eval_proof_acc(model, prover=dsp_magnus_hammer, formal_dataset_val, metrics=(proof_accuracy, ppl, exact_str_match, avg_acc, token_edit_distance))  # soft metrics (not proof acc) to see if we get a weaker signal from other metrics on target fl_gt
    eval_maf_proof_acc(model, prover=dsp_magnus_hammer, textbook=sipser_complexity_theory, metrics=(proof_accuracy, ppl, exact_str_match, avg_acc, token_edit_distance))  # proof accuracy checks if the (autoformalized) formal theorem (the human/gpt4 checked) is proved. But the soft/nlp metrics  are used on the informal proof from the textbook to see if model improves the prediction on at least that