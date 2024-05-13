import styled from "styled-components";

export const Container = styled.div`
  display: flex;
  align-items: center;
  background-color: #80c8b6;
  min-height: 100vh;
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

export const CardContainer = styled.div`
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  max-width: 120rem;
  flex-wrap: wrap;
  z-index: 1;
  padding-bottom: 15rem;
`;

export const Card = styled.div`
  position: relative;
  width: 28rem;
  height: 40rem;
  margin: 3rem;
  box-shadow: 2rem 2rem 5rem rgba(0, 0, 0, 0.5);
  border-radius: 1.5rem;
  background: rgba(255, 255, 255, 0.1);
  overflow: hidden;
  display: flex;
  justify-content: flex-start;
  justify-content: center;
  border-top: 0.1rem solid rgba(255, 255, 255, 05);
  border-left: 0.1rem solid rgba(255, 255, 255, 05);

  &:hover {
    transform: scale(1.03);
  }
`;

export const Content = styled.div`
  padding: 2rem;
  text-align: center;
  flex-direction: column;
  display: flex;
  gap: 1rem;
`;

export const H3 = styled.h3`
  font-size: 2.5rem;
  color: #fff;
  z-index: 1;
  font-family: "Platypi";
`;

export const P = styled.p`
  font-size: 2rem;
  color: #fff;
  font-weight: 300;
  font-family: "Platypi";
`;

export const ErrorModal = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem 1.6rem;
  gap: 1.2rem;
  background: rgba(255, 100, 100, 0.8);
  border-radius: 1.5rem;
  border-top: 0.1rem solid rgba(255, 255, 255, 05);
  border-left: 0.1rem solid rgba(255, 255, 255, 05);
  color: white;
  font-size: 2rem;
  font-family: "Platypi";

  @media (max-width: 561px) {
    width: 80%;
  }

  @media (max-width: 355px) {
    width: 90%;
  }
`;

export const ModalButton = styled(Button)`
  width: 60%;
  font-size: 2rem;
  background-color: #cc1717;

  &:hover {
    background-color: #f17a7a;
  }

  @media (max-width: 726px) {
    width: 80%;
  }
`;
