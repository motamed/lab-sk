# Copyright (c) Microsoft. All rights reserved.

from typing import Annotated, Any

from pydantic import BaseModel

from semantic_kernel.connectors.ai.open_ai import OpenAIEmbeddingPromptExecutionSettings
from semantic_kernel.data import (
    VectorStoreRecordDataField,
    VectorStoreRecordKeyField,
    VectorStoreRecordVectorField,
    vectorstoremodel,
)

###
# The data model used for this sample is based on the hotel data model from the Azure AI Search samples.
# When deploying a new index in Azure AI Search using the import wizard you can choose to deploy the 'hotel-samples'
# dataset, see here: https://learn.microsoft.com/en-us/azure/search/search-get-started-portal.
# This is the dataset used in this sample with some modifications.
# This model adds vectors for the 2 descriptions in English and French.
# Both are based on the 1536 dimensions of the OpenAI models.
# You can adjust this at creation time and then make the change below as well.
###


@vectorstoremodel
class HotelSampleClass(BaseModel):
    HotelId: Annotated[str, VectorStoreRecordKeyField]
    HotelName: Annotated[str | None, VectorStoreRecordDataField()] = None
    Description: Annotated[
        str,
        VectorStoreRecordDataField(
            has_embedding=True, embedding_property_name="description_vector", is_full_text_searchable=True
        ),
    ]
    description_vector: Annotated[
        list[float] | None,
        VectorStoreRecordVectorField(
            dimensions=1536,
            local_embedding=True,
            embedding_settings={"embedding": OpenAIEmbeddingPromptExecutionSettings(dimensions=1536)},
        ),
    ] = None
    Description_fr: Annotated[
        str, VectorStoreRecordDataField(has_embedding=True, embedding_property_name="description_fr_vector")
    ]
    description_fr_vector: Annotated[
        list[float] | None,
        VectorStoreRecordVectorField(
            dimensions=1536,
            local_embedding=True,
            embedding_settings={"embedding": OpenAIEmbeddingPromptExecutionSettings(dimensions=1536)},
        ),
    ] = None
    Category: Annotated[str, VectorStoreRecordDataField()]
    Tags: Annotated[list[str], VectorStoreRecordDataField()]
    ParkingIncluded: Annotated[bool | None, VectorStoreRecordDataField()] = None
    LastRenovationDate: Annotated[str | None, VectorStoreRecordDataField()] = None
    Rating: Annotated[float, VectorStoreRecordDataField()]
    Location: Annotated[dict[str, Any], VectorStoreRecordDataField()]
    Address: Annotated[dict[str, str | None], VectorStoreRecordDataField()]
    Rooms: Annotated[list[dict[str, Any]], VectorStoreRecordDataField()]