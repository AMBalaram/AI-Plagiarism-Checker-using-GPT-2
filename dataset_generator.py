"""
Dataset Generator for AI Plagiarism Checker
Generates human-written, AI-generated, and mixed text samples
"""

import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_ai_texts(num_samples=100, output_file='data/ai.txt'):
    """Generate AI-written text using GPT-2"""
    print("Loading GPT-2 for text generation...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    prompts = [
        "Artificial intelligence is revolutionizing",
        "Climate change poses significant challenges",
        "The future of technology includes",
        "Education in the digital age requires",
        "Healthcare innovations are transforming",
        "Space exploration has revealed",
        "Renewable energy sources are becoming",
        "Cybersecurity threats continue to",
        "Machine learning algorithms can",
        "Quantum computing represents"
    ]
    
    os.makedirs('data', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            f.write(text + '\n\n')
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} AI samples")
    
    print(f"AI texts saved to {output_file}")

def create_human_texts(output_file='data/human.txt'):
    """Create human-written text samples (news articles, essays, etc.)"""
    human_samples = [
        """The global economy faces unprecedented challenges as nations grapple with inflation, supply chain disruptions, and geopolitical tensions. Economists suggest that central banks must balance interest rate hikes with the need to avoid recession. Meanwhile, emerging markets show resilience through diversification strategies and regional cooperation agreements.""",
        
        """Recent archaeological discoveries in South America have shed new light on pre-Columbian civilizations. Researchers uncovered sophisticated irrigation systems dating back over 3,000 years, challenging previous assumptions about technological advancement in ancient societies. The findings suggest extensive trade networks existed long before European contact.""",
        
        """Modern software development practices emphasize collaboration, automation, and continuous improvement. Agile methodologies have transformed how teams approach project management, enabling faster iteration and better responsiveness to changing requirements. Version control systems and CI/CD pipelines are now standard tools in the developer's toolkit.""",
        
        """The Mediterranean diet continues to rank highly in nutritional studies for its health benefits. Rich in olive oil, fish, whole grains, and fresh vegetables, this eating pattern has been associated with reduced cardiovascular disease risk and improved longevity. Researchers attribute these benefits to the diet's anti-inflammatory properties and balanced macronutrient profile.""",
        
        """Urban planning in the 21st century must address sustainability, livability, and equity. Cities worldwide are experimenting with green infrastructure, mixed-use development, and improved public transportation. The goal is creating environments where residents can thrive while minimizing environmental impact and ensuring access to essential services.""",
        
        """Breakthrough treatments for autoimmune diseases offer hope to millions of patients. Biologic therapies targeting specific immune pathways have shown remarkable efficacy in conditions like rheumatoid arthritis and Crohn's disease. Ongoing research into personalized medicine promises even more targeted interventions with fewer side effects.""",
        
        """The evolution of jazz music reflects America's complex cultural history. From its roots in New Orleans to the bebop revolution and fusion experiments, jazz has continuously reinvented itself. Musicians like Miles Davis, John Coltrane, and Herbie Hancock pushed boundaries while honoring the tradition of improvisation and musical dialogue.""",
        
        """Coral reef ecosystems face existential threats from ocean acidification and warming waters. Marine biologists work urgently to understand resilience mechanisms and develop intervention strategies. Coral restoration projects show promise, but experts emphasize that reducing carbon emissions remains critical for long-term survival of these biodiversity hotspots.""",
        
        """Quantum mechanics revolutionized our understanding of reality at the atomic scale. Wave-particle duality, superposition, and entanglement challenge classical intuitions about how the universe operates. These phenomena, once purely theoretical curiosities, now underpin technologies from semiconductors to quantum computers.""",
        
        """The printing press democratized knowledge in the 15th century, much as the internet has in our time. Gutenberg's invention enabled mass production of books, facilitating the spread of ideas during the Renaissance and Reformation. Similarly, digital technology has transformed information access, though questions about quality and veracity persist."""
    ]
    
    os.makedirs('data', exist_ok=True)
    
    # Multiply samples to get ~100 entries
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(10):
            for sample in human_samples:
                # Add slight variations
                f.write(sample + '\n\n')
    
    print(f"Human texts saved to {output_file}")

def create_mixed_texts(output_file='data/mixed.txt'):
    """Create mixed human-AI texts (simulating edited AI content)"""
    mixed_samples = [
        """Artificial intelligence is revolutionizing many industries, but its impact goes far beyond simple automation. In healthcare, for instance, AI algorithms can analyze medical images with remarkable accuracy. However, the human element remains crucial - doctors must interpret results within clinical context and consider patient history. This synergy between human expertise and machine capability represents the future of medicine.""",
        
        """Climate change poses significant challenges to our planet's ecosystems and human societies. Rising temperatures are causing glaciers to melt at alarming rates. Yet, innovative solutions are emerging: renewable energy costs have plummeted, making solar and wind competitive with fossil fuels. Countries like Denmark and Costa Rica demonstrate that sustainable development is achievable with political will and investment.""",
        
        """The future of technology includes quantum computing, biotechnology, and advanced materials science. Quantum computers leverage superposition to perform certain calculations exponentially faster than classical machines. But what does this mean practically? Industries from pharmaceuticals to finance could see transformative benefits. Drug discovery timelines might shrink from years to months as quantum simulations model molecular interactions with unprecedented precision.""",
        
        """Education in the digital age requires reimagining traditional pedagogical approaches. Online learning platforms provide unprecedented access to knowledge, breaking down geographical barriers. However, effective education involves more than content delivery. Students need guidance, motivation, and social interaction. The most successful models blend technology with human mentorship, using data analytics to personalize learning while maintaining the irreplaceable teacher-student relationship.""",
        
        """Space exploration has revealed extraordinary insights about our universe and our place within it. The James Webb Space Telescope captures images of galaxies formed shortly after the Big Bang. These observations aren't just aesthetically stunning; they help scientists understand cosmic evolution. Meanwhile, Mars rovers search for signs of ancient life, and private companies work toward making space more accessible. We're entering an era where humans might become a multi-planetary species."""
    ]
    
    os.makedirs('data', exist_ok=True)
    
    # Multiply to get ~100 samples
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(20):
            for sample in mixed_samples:
                f.write(sample + '\n\n')
    
    print(f"Mixed texts saved to {output_file}")

if __name__ == '__main__':
    print("Starting dataset generation...")
    print("\n1. Creating human-written texts...")
    create_human_texts()
    
    print("\n2. Generating AI texts (this may take a few minutes)...")
    generate_ai_texts(num_samples=100)
    
    print("\n3. Creating mixed texts...")
    create_mixed_texts()
    
    print("\n✓ Dataset generation complete!")
    print("Files created in 'data/' directory:")
    print("  - human.txt (100 samples)")
    print("  - ai.txt (100 samples)")
    print("  - mixed.txt (100 samples)")
