from streamlit_feedback import streamlit_feedback
from app import ChatbotApp
import json
from sklearn.metrics import accuracy_score
import time

class ChatbotEvaluator:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.load_test_data()

    def load_test_data(self):
        # Load questions and answers from a JSON file
        with open('test_data.json', 'r') as f:
            self.test_data = json.load(f)

    def evaluate_accuracy(self):
        correct = 0
        total = len(self.test_data)
        for item in self.test_data:
            question = item['question']
            ideal_answer = item['answer']
            chatbot_answer = self.chatbot.conversation_chat(question)
            if self.compare_answers(chatbot_answer, ideal_answer):
                correct += 1
        return correct / total

    def compare_answers(self, generated, ideal):
        
        return all(keyword.lower() in generated.lower() for keyword in ideal.split())

    def measure_response_time(self):
        times = []
        for item in self.test_data:
            question = item['question']
            start_time = time.time()
            self.chatbot.conversation_chat(question)
            end_time = time.time()
            times.append(end_time - start_time)
        return sum(times) / len(times)

    def evaluate_consistency(self):
        
        topics = {}
        for item in self.test_data:
            topic = item.get('topic', 'general')
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(item['question'])

        consistencies = []
        for topic, questions in topics.items():
            responses = [self.chatbot.conversation_chat(q) for q in questions]
            consistencies.append(self.calculate_consistency(responses))
        return sum(consistencies) / len(consistencies)

    def calculate_consistency(self, responses):
        
        unique_responses = set(responses)
        return 1 - (len(unique_responses) - 1) / len(responses)

    def run_evaluation(self):
        accuracy = self.evaluate_accuracy()
        avg_response_time = self.measure_response_time()
        consistency_score = self.evaluate_consistency()

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Average Response Time: {avg_response_time:.2f} seconds")
        print(f"Consistency Score: {consistency_score:.2f}")

if __name__ == "__main__":
    chatbot = ChatbotApp()
    evaluator = ChatbotEvaluator(chatbot)
    evaluator.run_evaluation()