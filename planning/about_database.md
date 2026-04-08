# Goal of the database
The database is primarily for logging what is sent to the LLM, recieved from the LLM, what requests are placed with the HTTP service, and results for specific images run through the CV model.

# Schema
Contains some very long text fields, but that's okay since this is very small scale.
## images
Information about the training images and their correct classification, almost directly corresponds to driver_imgs_list.csv from the original dataset with the addition of an image_id
- PK image_id (int)- autoincrementing identifier
- image_name (varchar 50)- name of the image file
- class (int)- correct class of the image
- FK driver_id (int)- id of driver in image

## http 
Log of http requests made
- PK http_id (int)- autoincremementing identifier
- time_http_received (datetime)- time request received
- potentially add arguments column

## cv_results
- PK cv_id (int)- autoincrementing identifier
- FK http_id (int)- id of the http request that triggered this input
- FK image_id (int)- id of image that was sent to the model
- result (text)- CV's prediction resulting from the input

## llm
- PK llm_id (int)- autoincrementing identifier
- FK http_id (int)- http_id associated with
- time_sent (datetime)- timestamp request sent
- time_received (datetime) - timestamp response received from the LLM
- prompt_type (enum 'image', 'summary', 'other')- type of prompt- is it one of the 10 images, asking for the summary, or doing something else (probably for testing)?
- prompt_body (text)- body of the prompt that was sent to the llm through the code
- result (text)- entire response recieved from the LLM
