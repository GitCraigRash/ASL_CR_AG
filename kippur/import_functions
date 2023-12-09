def search_asl_images(directory, number_of_images):
  """
  Searches root directory until it finds dictionary containing images.
  Calls collect_training_images. 
  Returns dictionary with specified number of images.

  imputs:
  directory
  number_of_images

  outputs:
  dictionary of images as keys and labels as values
  """

  curr_file = os.listdir(directory)
  for item in curr_file:
    if item == 'A':
      image_arrays, image_labels = collect_training_images(directory, number_of_images)
    else:
      sub_directory = os.path.join(directory,item)
      image_arrays, image_labels = search_asl_images(sub_directory, number_of_images)
  return image_arrays, image_labels

def collect_training_images(directory, number_of_images):
  import re
  pattern = r'/([^/\d]+)(\d+)\.jpg$'
  #print(directory)
  image_arrays = []
  image_labels = []
  for letter in os.listdir(directory):
    letter_file = os.path.join(directory,letter)
    #print(letter_file)
    for number, image in enumerate(os.listdir(letter_file)):
    #print(number,os.path.join(directory,sub_file))
      if number == number_of_images - 1:
        break
      #print(number,image_directory)
      image_directory = os.path.join(letter_file,image)
      img = Image.open(image_directory)
      img_array = np.array(img)
      image_arrays.append(img_array)
      #print(image_arrays)
      matches = re.search(pattern, image_directory)
      label = matches.group(1)
      image_labels.append(label)

  image_arrays = np.array(image_arrays)
  image_labels = np.array(image_labels)
  return image_arrays, image_labels
