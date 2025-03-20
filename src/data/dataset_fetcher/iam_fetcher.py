import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import random
from src.data.utils.xml_utils import crawlerXML
from src.data.utils.alphabet import Alphabet

def _get_sample_names(root):
    xml_path = os.path.join(root,"xml")
    paragraph_path = os.path.join(root,"paragraphs")
    xml_file_names = [os.path.splitext(f)[0] for f in os.listdir(xml_path)]
    paragraph_file_names = [os.path.splitext(f)[0] for f in os.listdir(paragraph_path)]
    xml_file_names.sort()
    paragraph_file_names.sort()
    assert xml_file_names==paragraph_file_names
    return paragraph_file_names

def _get_sample_names_from_list(split_file):
    with open(split_file, "r") as f:
        entries = f.readlines()
    entries = [e.strip() for e in entries]
    return entries

def preload_all_iam(root, split, max_samples=-1,
                    mode="RGB", alphabet=None, TestRun= False, modeParagraph = 0,
                    **kwargs):
    if alphabet==None:
        alphabet=Alphabet(dataset="IAM", mode="both")
    sample_names = _get_sample_names_from_list(os.path.join(root,split)+".uttlist")
    if(TestRun):
        sample_names = _get_sample_names_from_list(os.path.join(root, split) + ".uttlistt")

    xml_root = os.path.join(root,"xml")
    images = dict()
    meta_data = dict()

    if modeParagraph == 0:
        paragraph_root = os.path.join(root, "paragraphs")
        ext = os.path.splitext(os.listdir(paragraph_root)[0])[1]




        for s in tqdm(sample_names):
            xml_data = _extract_data_from_xml(os.path.join(xml_root,s)+".xml")



            img = Image.open(os.path.join(paragraph_root,s)+ext).convert(mode)
            images[s] = img.crop((xml_data["paragraph"]["left"], xml_data["paragraph"]["top"],
                                  xml_data["paragraph"]["right"], xml_data["paragraph"]["bottom"]))
            meta_data[s] = {"text": xml_data["paragraph"]["text"],
                            "writer": xml_data["paragraph"]["writer"],
                            "text_logits": alphabet.string_to_logits(xml_data["paragraph"]["text"])}

        return sample_names, meta_data, images

    else:
        lines_root = os.path.join(root, "lines")
        ext = os.path.splitext(os.listdir(lines_root)[0])[1]
        sample_names_lines = []
        line_counter = modeParagraph

        for s in tqdm(sample_names):
            xml_data = _extract_data_from_xml(os.path.join(xml_root, s) + ".xml")
            id_directory = s[0:3]
            id_directory_path = os.path.join(lines_root, id_directory)
            img_directory = os.path.join(id_directory_path, s)
            list_of_all_files = os.listdir(img_directory)
            random.shuffle(list_of_all_files)

            amount_of_files = len(list_of_all_files)
            counter = 0

            while counter < amount_of_files:
                #TODO this might not work if we want to have >3 lines
                if counter + line_counter >= amount_of_files:
                    pic_name = list_of_all_files[counter]
                    image_path = os.path.join(img_directory, pic_name)
                    name = pic_name[:-4]
                    sample_names_lines.append(name)

                    images[name] = Image.open(image_path).convert(mode)#.resize((1024,64))
                    meta_data[name] = {"text": xml_data["line"][name]["text"],
                                       "writer": xml_data["line"][name]["writer"],
                                       "text_logits": alphabet.string_to_logits(xml_data["line"][name]["text"])}
                    counter = counter + 1
                else:
                    # lets combine multiple lines to 1 picture

                    name = ""
                    writer = 0
                    combined_text = ""

                    list_of_images = []
                    max_width = 0
                    max_height = 0
                    off_set_top_sides = 5

                    for i in range(line_counter):

                        pic_name = list_of_all_files[counter+i]

                        #get writer
                        if i == 0 :
                            #all should have the same writer. Hope
                            writer = xml_data["line"][pic_name[:-4]]["writer"]

                        #add newline and combine the text name
                        if i != 0:
                            combined_text = combined_text + '\n'
                            name = name + '&'

                        # combine Text
                        combined_text = combined_text + xml_data["line"][pic_name[:-4]]["text"]
                        # combine names
                        name = name + pic_name[:-4]

                        #open image
                        img = Image.open(os.path.join(img_directory, pic_name)).convert(mode)
                        list_of_images.append(img)

                        #set new values for height and width
                        max_width = max(max_width,img.width)
                        max_height = max_height + img.height

                    combined_img = Image.new(mode, (max_width+2*off_set_top_sides, max_height+2*off_set_top_sides),color='white')
                    curent_height = off_set_top_sides

                    for img in list_of_images:
                        combined_img.paste(img, (off_set_top_sides, curent_height))
                        curent_height = curent_height + img.height

                    sample_names_lines.append(name)
                    images[name] = combined_img
                    combined_logits = alphabet.string_to_logits(combined_text)
                    counter = counter + line_counter
                    meta_data[name] = {"text": combined_text,
                                       "writer": writer,
                                       "text_logits": combined_logits}

        return sample_names_lines, meta_data, images


def _extract_data_from_xml(file_path):
    xml = crawlerXML(file_path)
    name = os.path.split(os.path.splitext(file_path)[0])[1]
    writer = int(xml.find("form")["writer-id"])
    lines = xml.find_all("line")
    out_dict = {}
    for l in lines:
        id = l["id"]
        text = l["text"].replace("&quot;",'"').replace("&amp;", "&")
        id_parts = l["id"].split("-")
        # if "-----" in text:
        #     print(id, text)
        #     exit()
        out_dict[id] = {"text":text, "writer": writer, "paragraph": name}

    cmps = xml.find_all("cmp")
    left, right, top, bottom = [], [], [], []
    for c in cmps:
        left.append(int(c["x"]))
        right.append(int(c["x"]) + int(c["width"]))
        top.append(int(c["y"]))
        bottom.append(int(c["y"]) + int(c["height"]))
    paragraph_data = _min_max_coords(left, right, top, bottom)
    paragraph_data["writer"] = writer
    paragraph_data["text"] = _assemble_text(out_dict)
    paragraph_dict = {name: paragraph_data}
    return {"line": out_dict, "paragraph": paragraph_data}

def _assemble_text(line_dict):
    text = ""
    keys = [int(k.split("-")[-1]) for k in line_dict.keys()]
    keys.sort()
    l = list(line_dict.keys())[0].split("-")[:-1]
    paragraph = "-".join(l)
    for k in keys:
        k = paragraph + "-" +str(k).zfill(2)
        text+=line_dict[k]["text"]+"\n"

    return text

def _min_max_coords(left, right, top, bottom):
    assert len(left) == len(right) == len(top) == len(bottom), "{} {} {} {}".format(len(left), len(right), len(top), len(bottom))
    left_min = np.min(np.asarray(left))
    right_max = np.max(np.asarray(right))
    top_min = np.min(np.asarray(top))
    bottom_max = np.max(np.asarray(bottom))
    return {"left": left_min, "right": right_max,
            "top": top_min, "bottom": bottom_max}


def load_other_sources(root,split, paragraphs=True):

    alphabet=Alphabet(dataset="IAM", mode="both")
    sample_names = _get_sample_names_from_list(os.path.join(root,split)+".uttlist")

    xml_root = os.path.join(root,"xml")
    images = dict()
    meta_data = dict()

    if paragraphs:
        paragraph_root = os.path.join(root, "paragraphs")
        ext = os.path.splitext(os.listdir(paragraph_root)[0])[1]

        for s in tqdm(sample_names):
            xml_data = _extract_data_from_xml(os.path.join(xml_root,s)+".xml")

            img = Image.open(os.path.join(paragraph_root,s)+ext).convert('L')
            images[s] = img

            meta_data[s] = {"text": xml_data["paragraph"]["text"],
                                "writer": xml_data["paragraph"]["writer"],
                                "text_logits": alphabet.string_to_logits(xml_data["paragraph"]["text"])}

        return sample_names, meta_data, images

    lines_root = os.path.join(root, "lines")
    ext = os.path.splitext(os.listdir(lines_root)[0])[1]
    sample_names_lines = []

    for s in tqdm(sample_names):
        xml_data = _extract_data_from_xml(os.path.join(xml_root, s) + ".xml")
        id_directory = s[0:3]
        id_directory_path = os.path.join(lines_root, id_directory)
        img_directory = os.path.join(id_directory_path, s)
        list_of_all_files = os.listdir(img_directory)
        random.shuffle(list_of_all_files)

        amount_of_files = len(list_of_all_files)
        counter = 0

        while counter < amount_of_files:
            # TODO this might not work if we want to have >3 lines
            pic_name = list_of_all_files[counter]
            image_path = os.path.join(img_directory, pic_name)
            name = pic_name[:-4]
            sample_names_lines.append(name)

            images[name] = Image.open(image_path).convert('L')  # .resize((1024,64))
            meta_data[name] = {"text": xml_data["line"][name]["text"],
                                   "writer": xml_data["line"][name]["writer"],
                                   "text_logits": alphabet.string_to_logits(xml_data["line"][name]["text"])}
            counter = counter + 1




    return sample_names_lines, meta_data, images

if __name__ == "__main__":

    import h5py
    import uuid
