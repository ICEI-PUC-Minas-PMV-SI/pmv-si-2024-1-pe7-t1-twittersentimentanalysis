import axios, { AxiosResponse } from "axios";

export interface SentimentResponse {
  [key: string]: {
    prediction: string;
    probabilities: number;
  };
}

export const analysisSentiment = async (
  input: string
): Promise<SentimentResponse> => {
  try {
    const payload = {
      "text": input,
    };

    const response: AxiosResponse<SentimentResponse> = await axios.post(
      "https://puc-analysis-sentiment.azurewebsites.net/predict",
      payload
    );

    console.log("Resposta da requisição:", response.data);

    return response.data;
  } catch (error) {
    console.error("Ocorreu um erro:", error);
    throw error;
  }
};
