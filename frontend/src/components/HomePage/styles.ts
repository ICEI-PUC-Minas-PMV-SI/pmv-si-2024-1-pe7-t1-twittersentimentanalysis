import styled from "styled-components";

export const Container = styled.div`
  display: flex;
  align-items: center;
  background-color: #80c8b6;
  height: 100vh;
  padding-top: 7rem;
  flex-direction: column;
  gap: 2rem;
`;

export const Image = styled.img`
  max-height: 30rem;
  max-width: 30rem;

  @media (max-width: 468px) {
    max-height: 28rem;
    max-width: 28rem;
  }
`;

export const Input = styled.input`
  background-color: #feebd4;
  border-radius: 1rem;
  border-color: #351b00;
  color: #351b00;
  font-family: "Platypi";
  font-size: 1.5rem;
  width: 100rem;
  height: 5rem;
  padding-left: 2%;

  @media (max-width: 1120px) {
    width: 80%;
  }
  @media (max-width: 468px) {
    width: 95%;
  }
`;

export const Button = styled.button`
  width: 30rem;
  background-color: #351b00;
  color: white;
  font-family: "Platypi";
  font-size: 2.5rem;
  border-radius: 2rem;
  border: 0.5rem double white;
  cursor: pointer;
  padding: 0.5rem;
  transition: background-color 0.3s;
  display: flex;
  justify-content: center;

  &:hover {
    background-color: #454342;
  }

  &:active {
    transform: scale(1.1);
  }

  @media (max-width: 700px) {
    width: 40%;
  }
  @media (max-width: 468px) {
    width: 95%;
    font-size: 2rem;
  }
`;

export const Spinner = styled.div`
  width: 2.5rem;
  height: 2.5rem;
  border: 0.5rem solid #80c8b6;
  border-top-color: #feebd4;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

export const Span = styled.span`
  font-family: "Platypi";
  font-size: 4rem;
  text-align: justify;

  @media (max-width: 954px) {
    text-align: center;
  }

  @media (max-width: 482px) {
    font-size: 3rem;
  }
`;

export const SentimentResultPositive = styled(Span)`
  background-color: green;
  border-radius: 0.5rem;
`;

export const SentimentResultNegative = styled(SentimentResultPositive)`
  background-color: black;
  color: #ffffff;
`;

export const SentimentResultLitigious = styled(SentimentResultPositive)`
  background-color: red;
`;

export const SentimentResultUncertainty = styled(SentimentResultPositive)`
  background-color: #e9f4e0;
`;
