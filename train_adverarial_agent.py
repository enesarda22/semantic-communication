from tqdm import tqdm

import torch
from torch.nn import functional as F

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.model.adversarial_agent import AdversarialAgent
from simple_train import print_loss


def negative_mse_loss(output, target) -> torch.Tensor:
    return -F.mse_loss(output, target)


if __name__ == "__main__":
    device = torch.device("cpu")

    data_handler = DataHandler(device, data_size=40000)
    data_handler.load_data(with_encoder_output=True)

    agent = AdversarialAgent(n_embeddings=384)
    agent_optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4)

    relay = AdversarialAgent(n_embeddings=384)
    relay_optimizer = torch.optim.AdamW(relay.parameters(), lr=1e-4)

    for _ in range(15):
        train_losses = []
        agent.train()
        relay.train()

        for b in tqdm(data_handler.train_dataloader):
            encoder_output = b[1].to(device)

            agent_output = agent(encoder_output)
            # relay_output = relay(encoder_output)

            # train agent
            agent_loss = negative_mse_loss(agent_output, encoder_output)
            agent_optimizer.zero_grad()
            agent_loss.backward()
            agent_optimizer.step()

            # train relay
            # relay_loss = F.mse_loss(
            #     agent_output.detach() + relay_output, encoder_output
            # )
            # relay_optimizer.zero_grad()
            # relay_loss.backward()
            # relay_optimizer.step()
            #
            train_losses.append(agent_loss.item())

        val_losses = []
        agent.eval()
        relay.eval()
        for b in data_handler.val_dataloader:
            encoder_output = b[1].to(device)

            agent_output = agent(encoder_output)
            # relay_output = relay(encoder_output)

            # superposed_output = agent_output.detach() + relay_output.detach()
            # loss = F.mse_loss(superposed_output, encoder_output)

            loss = negative_mse_loss(agent_output + encoder_output, encoder_output)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")
