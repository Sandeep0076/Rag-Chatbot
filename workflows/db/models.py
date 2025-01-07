class User(BaseModel):
    id: str
    email: str
    name: str
    wf_deletion_candidate: bool = default(False)
    wf_deletion_timestamp: Optional[datetime]
    conversations: List[Conversation]
    folders: List[Folder]


class Message(BaseModel):
    id: str
    role: str
    content: str
    createdAt: datetime = Field(default_factory=datetime.now(timezone.utc))
    conversation: Any
    conversationId: str


class Conversation(BaseModel):
    id: str
    user: Any
    messages: List[Message]
    temperature: int
    folder: Optional[Any]
    folderId: Optional[str]
    prompt: str
    model: Optional[Any]
    modelId: str
    name: str
    userEmail: str
    updatedAt: datetime = Field(default_factory=datetime.now(timezone.utc))
    createdAt: datetime = Field(default_factory=datetime.now(timezone.utc))


class Model(BaseModel):
    id: str
    name: str
    conversations: List[Conversation]
    maxLength: int
    tokenLimit: int


class Folder(BaseModel):
    id: str
    user: Any
    userId: str
    name: str
    isRoot: bool = default(False)
    conversations: List[Conversation]
