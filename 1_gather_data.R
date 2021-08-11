library(tidyverse)
library(janitor)
library(tools)
library(reshape2)
library(groupdata2)
library(fs)


data_dir <- file.path('..', '..', 'irop_data')

irop_data <- file.path(data_dir, 'irop_07092020.csv') %>%
    read_csv(col_types = cols())

fundus_images <- file.path(data_dir, 'retcam') %>%
    list.files() %>%
    as.data.frame() %>%
    mutate(image_id = file_path_sans_ext(.)) %>%
    rename(fundus_location = '.')

segmentations <- file.path(data_dir, 'segmentations') %>%
    list.files() %>%
    as.data.frame() %>%
    mutate(image_id = file_path_sans_ext(.)) %>%
    rename(segmentation_location = '.')

common_images <- fundus_images %>%
    inner_join(segmentations, by = 'image_id')


melted_data <- irop_data %>%
    clean_names() %>%
    filter(super_ethnicity == 'Non-Hispanic',
           race != 'American Indian or Alaska Native',
           race != 'Native Hawaiian or Pacific Islander, Unknown Type',
           race != 'Middle Eastern') %>%
    select(subject_id, race, posterior, inferior, superior, nasal, temporal) %>%
    melt(id.vars = c('subject_id', 'race')) %>%
    na_if('NULL') %>%
    drop_na() %>%
    distinct() %>%
    mutate(image_id = file_path_sans_ext(basename(value)),
           race = case_when(race == 'Caucasian/White' ~ 'white',
                            race == 'African American' ~ 'black',
                            race == 'Other Asian -> Specify' ~ 'asian',
                            race == 'Korean' ~ 'asian',
                            race == 'Vietnamese' ~ 'asian', 
                            race == 'Chinese' ~ 'asian',
                            race == 'Asian Indian' ~ 'asian', 
                            race == 'Filipino' ~ 'asian',
                            race == 'Japanese' ~ 'asian',
                            race == 'Thai' ~ 'asian', 
                            race == 'Asian, Unknown Type' ~ 'asian'
                            )) %>%
    inner_join(common_images, by = 'image_id') %>%
    mutate(across(c(subject_id, race, variable), as.factor))



set.seed(1337)

data_partitioned <- melted_data %>%
    partition(p = c(0.7, 0.15),
              id_col = 'subject_id',
              cat_col = 'race')

data_train <- data_partitioned[[1]] %>%
    balance(size = 'min', cat_col = 'race') %>%
    ungroup()

data_val <- data_partitioned[[2]] %>%
    balance(size = 'min', cat_col = 'race') %>%
    ungroup()

data_test <- data_partitioned[[3]] %>%
    balance(size = 'min', cat_col = 'race') %>%
    ungroup()


write_csv(data_train, 'out/train_data.csv')
write_csv(data_val, 'out/val_data.csv')
write_csv(data_test, 'out/test_data.csv')

dir_create('./out/retcam/train/asian')
dir_create('./out/retcam/train/black')
dir_create('./out/retcam/train/white')

dir_create('./out/retcam/val/asian')
dir_create('./out/retcam/val/black')
dir_create('./out/retcam/val/white')

dir_create('./out/retcam/test/asian')
dir_create('./out/retcam/test/black')
dir_create('./out/retcam/test/white')

dir_create('./out/segmentations/train/asian')
dir_create('./out/segmentations/train/black')
dir_create('./out/segmentations/train/white')

dir_create('./out/segmentations/val/asian')
dir_create('./out/segmentations/val/black')
dir_create('./out/segmentations/val/white')

dir_create('./out/segmentations/test/asian')
dir_create('./out/segmentations/test/black')
dir_create('./out/segmentations/test/white')


src = '/Volumes/External/irop_data/retcam'

dst = './out/retcam/train/'
file_copy(paste(src, data_train$fundus_location, sep = '/'),
          paste(dst, data_train$race, data_train$fundus_location, sep = '/'),
          overwrite = TRUE)

dst = './out/retcam/val/'
file_copy(paste(src, data_val$fundus_location, sep = '/'),
          paste(dst, data_val$race, data_val$fundus_location, sep = '/'),
          overwrite = TRUE)

dst = './out/retcam/test/'
file_copy(paste(src, data_test$fundus_location, sep = '/'),
          paste(dst, data_test$race, data_test$fundus_location, sep = '/'),
          overwrite = TRUE)


src = '/Volumes/External/irop_data/segmentations'

dst = './out/segmentations/train/'
file_copy(paste(src, data_train$segmentation_location, sep = '/'),
          paste(dst, data_train$race, data_train$segmentation_location, sep = '/'),
          overwrite = TRUE)

dst = './out/segmentations/val/'
file_copy(paste(src, data_val$segmentation_location, sep = '/'),
          paste(dst, data_val$race, data_val$segmentation_location, sep = '/'),
          overwrite = TRUE)

dst = './out/segmentations/test/'
file_copy(paste(src, data_test$segmentation_location, sep = '/'),
          paste(dst, data_test$race, data_test$segmentation_location, sep = '/'),
          overwrite = TRUE)
