# -*- coding: utf-8 -*-

"""
13_train_esm2_global_baseline.py


ESM2-GP baseline

Frozen ESM2 protein-level embedding
+
Symmetric Pair Operator
+
MLP classifier


Used for paper:
ESM2-GP baseline comparison


Input:
Data/embeddings/ESM2_t33/*.pt

Each file:
protein_id.pt

Tensor:
[1280]


Output:
Results/esm2_global_baseline/


"""


from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


import torch
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader



# ============================================================
# Path
# ============================================================


ROOT = Path(
    str(Path(__file__).resolve().parents[1])
)


DATA_DIR = (
    ROOT
    /
    "Data"
)


CANONICAL_DIR = (
    DATA_DIR
    /
    "processed"
    /
    "canonical"
)


# ============================
# IMPORTANT
# protein-level ESM2 embedding
# ============================

EMB_DIR = (
    DATA_DIR
    /
    "embeddings"
    /
    "ESM2_t33"
)



SAVE_DIR = (
    ROOT
    /
    "Results"
    /
    "esm2_global_baseline"
)


SAVE_DIR.mkdir(
    exist_ok=True,
    parents=True
)



# ============================================================
# Config
# ============================================================


SEEDS = [

    42,
    123,
    456,
    789,
    1234

]


DEVICE = (

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


EPOCHS = 100


BATCH_SIZE = 64


LR = 2e-4


WEIGHT_DECAY = 5e-4


PATIENCE = 15



# ============================================================
# seed
# ============================================================


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)



# ============================================================
# Dataset
# ============================================================


class BernettDataset(Dataset):


    def __init__(self,csv_file):


        self.df = pd.read_csv(
            csv_file
        )


        print(
            f"Loaded {csv_file.name}: {len(self.df)}"
        )



    def __len__(self):

        return len(self.df)



    def load_embedding(self,pid):


        path = (

            EMB_DIR
            /
            f"{pid}.pt"

        )


        if not path.exists():

            raise FileNotFoundError(

                f"Missing embedding:\n{path}"

            )


        emb = torch.load(

            path,

            map_location="cpu"

        )


        # support dict format

        if isinstance(emb,dict):


            if "embedding" in emb:

                emb = emb["embedding"]


            elif "representations" in emb:

                emb = emb["representations"]



        emb = emb.float()



        if emb.ndim != 1:

            raise RuntimeError(

                f"{pid}: expected [1280], got {emb.shape}"

            )



        if emb.shape[0] != 1280:

            raise RuntimeError(

                f"{pid}: expected 1280 dim, got {emb.shape}"

            )


        return emb



    def __getitem__(self,index):


        row = self.df.iloc[index]



        # protein columns

        # protein columns

        if "protein_a" in row.index:

            pa = row["protein_a"]
            pb = row["protein_b"]


        elif "protein_A" in row.index:

            pa = row["protein_A"]
            pb = row["protein_B"]


        elif "protein1" in row.index:

            pa = row["protein1"]
            pb = row["protein2"]


        elif "p1" in row.index:

            pa = row["p1"]
            pb = row["p2"]


        else:

            raise RuntimeError(
                f"Cannot find protein columns: {row.index}"
            )



        # label

        if "label" in row.index:

            y=row["label"]

        elif "interaction" in row.index:

            y=row["interaction"]

        elif "y" in row.index:

            y=row["y"]

        else:

            raise RuntimeError(

                f"Cannot find label column: {row.index}"

            )



        ea=self.load_embedding(pa)

        eb=self.load_embedding(pb)



        return (

            ea,

            eb,

            torch.tensor(

                y,

                dtype=torch.float32

            )

        )




# ============================================================
# Model
# ============================================================


class ESM2GlobalBaseline(nn.Module):


    def __init__(self):

        super().__init__()



        self.project = nn.Sequential(


            nn.Linear(

                1280,

                384

            ),


            nn.LayerNorm(

                384

            ),


            nn.GELU(),


            nn.Dropout(

                0.35

            )


        )



        self.classifier = nn.Sequential(


            nn.Linear(

                384*3,

                256

            ),


            nn.GELU(),


            nn.Dropout(

                0.35

            ),


            nn.Linear(

                256,

                1

            )


        )




    def forward(self,a,b):


        a=self.project(a)

        b=self.project(b)



        pair=torch.cat(

            [


                a+b,


                torch.abs(a-b),


                a*b


            ],

            dim=1

        )


        logit=self.classifier(pair)


        return logit.squeeze(1)




# ============================================================
# train
# ============================================================


def train_one_seed(seed):


    print("="*80)

    print(
        f"Training seed {seed}"
    )

    print("="*80)


    set_seed(seed)



    train_dataset=BernettDataset(

        CANONICAL_DIR
        /
        "train_pairs.csv"

    )


    val_dataset=BernettDataset(

        CANONICAL_DIR
        /
        "val_pairs.csv"

    )



    train_loader=DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0

    )


    val_loader=DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )



    model=ESM2GlobalBaseline()

    model.to(DEVICE)



    optimizer=torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY

    )



    criterion=nn.BCEWithLogitsLoss()



    from sklearn.metrics import average_precision_score



    best_aucprc=0

    wait=0



    save_path=(

        SAVE_DIR

        /

        f"best_seed_{seed}.pt"

    )



    for epoch in range(EPOCHS):


        model.train()


        for a,b,y in train_loader:


            a=a.to(DEVICE)

            b=b.to(DEVICE)

            y=y.to(DEVICE)



            logit=model(a,b)


            loss=criterion(

                logit,

                y

            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



        # validation

        model.eval()


        probs=[]

        labels=[]


        with torch.no_grad():


            for a,b,y in val_loader:


                a=a.to(DEVICE)

                b=b.to(DEVICE)


                logit=model(a,b)


                probs.extend(

                    torch.sigmoid(logit)

                    .cpu()

                    .numpy()

                )


                labels.extend(

                    y.numpy()

                )



        val_aucprc=average_precision_score(

            labels,

            probs

        )


        print(

            f"Epoch {epoch:03d} "

            f"Val AUPRC={val_aucprc:.5f}"

        )



        if val_aucprc > best_aucprc:


            best_aucprc=val_aucprc

            wait=0


            torch.save(

                {

                    "model":

                    model.state_dict(),


                    "seed":

                    seed,


                    "best_val_auprc":

                    best_aucprc

                },

                save_path

            )


        else:

            wait+=1



        if wait>=PATIENCE:

            print(
                "Early stopping"
            )

            break



    return best_aucprc




# ============================================================
# main
# ============================================================


def main():


    results={}


    for seed in SEEDS:


        best=train_one_seed(seed)


        results[str(seed)]={

            "best_val_auprc":

            best

        }



    with open(

        SAVE_DIR
        /
        "summary.json",

        "w"

    ) as f:


        json.dump(

            results,

            f,

            indent=2

        )


    print("\nFinished")




if __name__=="__main__":

    main()