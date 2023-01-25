#The goal of this file is to expose the custom model that we created as an FastAPI
from fastapi import FastAPI,Request
from transformers import BertTokenizer, BertForSequenceClassification


def get_model_from_hub():
    tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('rahulpointer/bertmodel_custom')
    return tokenizer, model
    

#Calling the method to get the tokenizer and the model.
tokenizer, model = get_model_from_hub()


app = FastAPI()

#Asynchronous post method takes request type input and from there data for the post request can be extracted.
@app.post('/predict')
async def predict_text(request: Request):

    data_post = await request.json()

    if text_input in data_post:
        user_input = data_post['text']
        test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
        output = model(**test_sample)
        y_pred = np.argmax(output.logits.detach().numpy(),axis=1)  
        response = {"Recieved Text": user_input,"Prediction": d[y_pred[0]]}

    else:
        response = {'result':'No text data provided or invalid key'}
        return response


#Starts from here the main function.
if __name__ == "__main__":
    #run -- filename.app object , localhost, port, reload=True means it reloads everytime the file is updated.
    uvicorn.run("main:app", host='0.0.0.0', port=8501, reload=True, debug=True)
