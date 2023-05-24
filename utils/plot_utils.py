def extract_crossval_results(results:dict, var:str, new_var:str)->dict:
    extracted_results = dict()
    extracted_results[new_var]=[]
    extracted_results['fold']=[]
    extracted_results['epoch']=[]
    cpt_epoch = 0
    for fold in results:
        extracted_list = [t.cpu().numpy()[()] for t in results[fold][var]]
        extracted_fold = [fold for i in range(len(extracted_list))]
        extracted_results[new_var] += extracted_list
        extracted_results['fold'] += extracted_fold
        extracted_results['epoch'] += [i-cpt_epoch for i in range(len(extracted_list))]
    return extracted_results