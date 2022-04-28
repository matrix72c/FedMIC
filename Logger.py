import pandas as pd


def init_logger():
    client_loss_df = pd.DataFrame(columns=['client_id', 'local_epoch', 'loss'])
    client_loss_df.to_csv("./result/client_loss.csv", mode='w')

    distill_result_df = pd.DataFrame(columns=['round', 'distill_loss', 'hr', 'hdcg'])
    distill_result_df.to_csv("./result/distill_result.csv", mode='w')


def log_client_loss(client_id, local_epoch, loss):
    client_loss_df = pd.DataFrame([[client_id, local_epoch, loss]], columns=['client_id', 'local_epoch', 'loss'])
    client_loss_df.to_csv("./result/client_loss.csv", mode='a', header=False)


def log_distill_result(rnd, distill_loss, hr, hdcg):
    distill_result_df = pd.DataFrame([[rnd, distill_loss, hr, hdcg]], columns=['round', 'distill_loss', 'hr', 'hdcg'])
    distill_result_df.to_csv("./result/distill_result.csv", mode='a', header=False)
