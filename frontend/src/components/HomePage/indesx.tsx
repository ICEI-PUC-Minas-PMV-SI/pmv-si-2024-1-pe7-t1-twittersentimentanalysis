import * as S from "./styles";
import { useState } from "react";

export const HomePage = () => {
  const [sentimentResult, setSentimentResult] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  return (
    <S.Container>
      <S.Image
        src="src/assets/sentiment-analysis-logo.png"
        alt="logo-sentiment-analysis"
      />
      <S.Input
        type="text"
        name="input-search"
        placeholder="Insira sua expressão aqui..."
        onChange={() => {
          setSentimentResult(false);
        }}
      />
      <S.Button
        type="submit"
        onClick={() => {
          setSentimentResult(true);
          setIsLoading(true);
        }}
      >
        {isLoading ? <S.Spinner /> : "Analisar"}
      </S.Button>
      {sentimentResult && (
        <S.Span>
          O sentimento presente em sua frase é:{""}
          <S.SentimentResultPositive>positivo</S.SentimentResultPositive>
        </S.Span>
      )}
    </S.Container>
  );
};
