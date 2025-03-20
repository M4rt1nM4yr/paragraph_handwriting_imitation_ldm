# from src.data.dataset_fetcher.dummy_fetcher import preload_all_dummy, preload_meta_dummy, preload_clustering_dummy
from src.data.dataset_fetcher.iam_fetcher import preload_all_iam
# from src.data.dataset_fetcher.nbb_fetcher import preload_all_nbb, preload_meta_nbb, preload_clustering_nbb
# from src.data.dataset_fetcher.rimes_fetcher import preload_all_rimes, preload_meta_rimes, preload_clustering_rimes
import h5py
from src.data.utils.alphabet import Alphabet

def fetch_dataset(dataset_type, preload=True, **kwargs):
    if dataset_type.lower() == "IAM".lower():
        if preload:
            return preload_all_iam(**kwargs)
        raise NotImplementedError
    if dataset_type.lower() == "Synthetic".lower():
        return get_synthetic_Data(**kwargs)
    if dataset_type.lower() == "combined".lower():
        real_names, real_meta, real_images = preload_all_iam(**kwargs)
        synthetic_names, synthetic_meta, synthetic_images = get_synthetic_Data(**kwargs)
        print(" loaded and ready to combine",flush=True)


        synthetic_names = real_names + synthetic_names
        for name in real_names:
            synthetic_images[name] = real_images[name]
            synthetic_meta[name] = real_meta[name]

        return synthetic_names,synthetic_meta,synthetic_images

    return None

def get_synthetic_Data(dataset_file,alphabet,size,invert_image=False,**kwargs):
    if alphabet==None:
        alphabet=Alphabet(dataset="IAM", mode="both")

    dataset = h5py.File(dataset_file, "r")
    max_samples = len(dataset["images"])

    sample_names = []
    meta_data = dict()
    images = dict()

    for i in range(max_samples):

        name = str(i)
        text = dataset["labels"][i].decode("utf-8")
        writer = int(dataset["writer"][i].decode("utf-8"))

        meta_data[name] = dict()

        #insert meta_data

        meta_data[name]["text"] = text
        meta_data[name]["text_logits"] = alphabet.string_to_logits(text)
        meta_data[name]["writer"] = writer

        #add name
        sample_names.append(name)

        #add image
        images[name] = dataset["images"][i].reshape(size)
        if invert_image:
            images[name] = 255 - images[name]

    return sample_names, meta_data, images



    # elif dataset_type.lower() == "dummy".lower():
    #     if preload:
    #         return preload_all_dummy(**kwargs)
    #     return preload_meta_dummy(**kwargs)
    # elif dataset_type.lower() == "nbb".lower():
    #     if preload:
    #         return preload_all_nbb(**kwargs)
    #     return preload_meta_nbb(**kwargs)
    # elif dataset_type.lower() == "rimes".lower():
    #     if preload:
    #         return preload_all_rimes(**kwargs)
    #     return preload_meta_rimes(**kwargs)
