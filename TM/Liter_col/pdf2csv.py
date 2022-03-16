"""
 coding:utf-8
 Program function: Extract pdf file content and convert it to txt format content

"""
import fnmatch
import os
from win32com import client as wc
import pathlib
import csv
from tqdm import tqdm
"""
Function description: PDF file convert to txt, save in the root directory by default, support customization
Parameter description: 1. filePath: file path; 2. savePath: save path
"""
"""
Implementation steps:
     Define file path and dump path: split
     Modify the new file name: fnmatch
     Set the full save path: join
     Start application format conversion: Dispatch
     Save text: SaveAs
"""

def get_path_of_source(filename):
    p = pathlib.Path(filename)
    return p

def Pdf2Txt(filePath, savePath=r'F:\txt_file'):
    # 1、Split file path into file directory and file name
    dirs, filename = os.path.split(filePath)
    print('original file path：', dirs)
    print('original file name：', filename)
    # 2、Modify the suffix of the file
    # new_name = ""
    if fnmatch.fnmatch(filename, '*.pdf') or fnmatch.fnmatch(filename, '*PDF'):
        new_name = filename[:-4] + '.txt'  # Update file extension
    else:
        print('Incorrect format, only pdf format is supported')
        return
    # 3、Set a new file save path
    if savePath == '':
        savePath = dirs
    else:
        savePath = savePath
    pdf2txtPath = os.path.join(savePath, new_name)
    print('new file path=', pdf2txtPath)
    # 4、Load handler for text extraction，pdf->txt
    wordapp = wc.Dispatch('Word.Application')  # start the application
    mytxt = wordapp.Documents.Open(filePath)  # open file path
    # 5、Save text
    mytxt.SaveAs(pdf2txtPath, 4)  # Save in txt format, 4 represents the extracted text
    mytxt.Close()
    return pdf2txtPath


def txt_to_csv(filename):
    fileToRead = open(filename)
    file_text = fileToRead.read()
    with_our_added_commas = file_text.replace("\n", "")
    strings_without_inverted_commas = with_our_added_commas.replace("\"", "")
    fileToRead.close()
    return strings_without_inverted_commas


if __name__ == '__main__':
    with open('Extract_text.csv', 'a', newline='') as fout: #Saved csv file
        csv_writer = csv.writer(fout)
        data1 = ['Number', 'Filename', 'txt-1', 'txt-2', 'txt-3', 'txt-4', 'txt-5']
        csv_writer.writerow(data1)
    tables = ''
    filePath1 = 'F:\pdf2csv_test'   # Folder where the pdf file is located
    i = 1                           # Count
    for filename in tqdm(os.listdir(r'F:\pdf2csv_test')):
        try:
            filePath = os.path.join(filePath1, filename)
            new_filePath = Pdf2Txt(filePath)
            extract_data = txt_to_csv(new_filePath)
            with open('Extract_text.csv', 'a', newline='') as fout:
                csv_writer = csv.writer(fout)
                if len(extract_data) >= 32750 and len(extract_data) <= 65500:   #Since the cell can store up to 32767 characters, it needs to be separated
                    data1 = [i, filename, extract_data[:32750], extract_data[32750:], '', '', '']
                    csv_writer.writerow(data1)
                    i += 1
                elif len(extract_data) > 65500 and len(extract_data) <= 98250:
                    data1 = [i, filename, extract_data[:32750], extract_data[32750:65500], extract_data[65500:], '', '']
                    csv_writer.writerow(data1)
                    i += 1
                elif len(extract_data) > 98250 and len(extract_data) <= 131000:
                    data1 = [i, filename, extract_data[:32750], extract_data[32750:65500], extract_data[65500:98250], extract_data[98250:], '']
                    csv_writer.writerow(data1)
                    i += 1
                elif len(extract_data) > 131000:
                    data1 = [i, filename, extract_data[:32750], extract_data[32750:65500], extract_data[65500:98250], extract_data[98250:131000], extract_data[131000:163750], 'The amount of text is too large, it is recommended to check']
                    csv_writer.writerow(data1)
                    i += 1
                else:
                    data1 = [i, filename, extract_data, '', '', '', '']
                    csv_writer.writerow(data1)
                    i += 1
        except:
            with open('Handling_failed_literature.txt', 'a') as error_text:
                error_text.write(filename + '\n')


