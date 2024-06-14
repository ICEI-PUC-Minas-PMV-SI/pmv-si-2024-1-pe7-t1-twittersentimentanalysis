import {
  analysisSentiment,
  SentimentResponse,
} from "../../service/sentimentAnalysis";
import * as S from "./styles";
import { useState } from "react";

export const HomePage = () => {
  const [sentimentResult, setSentimentResult] = useState<SentimentResponse>({});
  const [isLoading, setIsLoading] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [error, setError] = useState(false);

  const mapToPortuguese = (result: string) => {
    switch (result) {
      case "positive":
        return "positivo";
      case "negative":
        return "negativo";
      case "litigious":
        return "litigioso";
      case "uncertainty":
        return "incerto";
      default:
        return result;
    }
  };

  const handleClick = async () => {
    setIsLoading(true);
    if (sentimentResult) {
      setSentimentResult({});
    }
    try {
      if (inputValue) {
        const response = await analysisSentiment(inputValue);
        const mappedResponse = Object.fromEntries(
          Object.entries(response).map(([key, value]) => {
            if (key !== "text") {
              return [
                key,
                { ...value, prediction: mapToPortuguese(value.prediction) },
              ];
            }
            return [key, value];
          })
        );

        setSentimentResult(mappedResponse);
      }
    } catch (error) {
      console.error("Ocorreu um erro:", error);
      setError(true);
    } finally {
      setIsLoading(false);
      setInputValue("");
    }
  };
  return (
    <S.Container>
      <S.Image
        src="/sentiment-analysis-logo.png"
        alt="logo-sentiment-analysis"
      />
      <S.Input
        type="text"
        name="input-search"
        placeholder="Insira sua expressão em inglês aqui..."
        value={inputValue}
        onChange={(e) => {
          setInputValue(e.target.value);
        }}
      />
      <S.Button type="submit" onClick={handleClick}>
        {isLoading ? <S.Spinner /> : "Analisar"}
      </S.Button>
      {sentimentResult && (
        <S.CardContainer>
          {Object.keys(sentimentResult)
            .filter((model) => model !== "text")
            .map((model, index) => (
              <S.Card key={index}>
                <S.Content>
                  <S.H3>{model}</S.H3>
                  <S.P>
                    <span>
                      De acordo com a análise do modelo {model}, o sentimento
                      presente na sua expressão é:{" "}
                    </span>
                    <strong>
                      {sentimentResult[model].prediction.toUpperCase()}!
                    </strong>
                  </S.P>
                  <S.P>
                    <strong>Probabilidade de acerto:</strong>{" "}
                    {Math.floor(sentimentResult[model].probabilities)}%.
                  </S.P>
                </S.Content>
              </S.Card>
            ))}
        </S.CardContainer>
      )}
      {error && (
        <S.ErrorModal>
          <span>Opa, algo deu errado :(</span>{" "}
          <span>Aguarde uns minutos e tente novamente!</span>
          <S.ModalButton type="button" onClick={() => setError(false)}>
            Tentar novamente
          </S.ModalButton>
        </S.ErrorModal>
      )}
    </S.Container>
  );
};
