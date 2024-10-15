import re

# map Prisma types to Pydantic types
prisma_to_pydantic_types = {
    "String": "str",
    "Int": "int",
    "Boolean": "bool",
    "DateTime": "datetime",
}

# map Prisma types to SQLAlchemy types
prisma_to_sqlalchemy_types = {
    "String": "String",
    "Int": "Integer",
    "Boolean": "Boolean",
    "DateTime": "DateTime",
}

# regex patterns to capture model and field information
model_regex = re.compile(r"model\s+(\w+)\s+\{([\s\S]+?)\}")
field_regex = re.compile(r"\s*(\w+)\s+(\w+\[\]|\w+)(\??)(.*)")


# function to convert Prisma schema to Pydantic models
def prisma_to_pydantic(schema: str) -> str:
    models = model_regex.findall(schema)
    pydantic_models = []

    for model_name, model_body in models:
        fields = field_regex.findall(model_body)
        model_lines = [f"class {model_name}(BaseModel):"]

        for field_name, field_type, is_optional, field_attributes in fields:
            # convert Prisma type to Python/Pydantic type
            pydantic_type = prisma_to_pydantic_types.get(field_type, "Any")

            # handle optional fields (marked with `?`)
            if is_optional:
                pydantic_type = f"Optional[{pydantic_type}]"

            # handle relation fields (like Post[], Conversation[], etc.)
            if field_type.endswith("[]"):
                relation_type = field_type[:-2]
                pydantic_type = f"List[{relation_type}]"

            # handle fields with default values (e.g., cuid(), now())
            if "@default(cuid())" in field_attributes:
                model_lines.append(
                    f"    {field_name}: {pydantic_type}"
                )  # = Field(default_factory=lambda: cuid())")
            elif "@default(now())" in field_attributes:
                model_lines.append(
                    f"    {field_name}: {pydantic_type} = Field(default_factory=datetime.now(timezone.utc))"
                )
            elif "@default(false)" in field_attributes:
                model_lines.append(
                    f"    {field_name}: {pydantic_type} = default(False)"
                )
            elif "@default(true)" in field_attributes:
                model_lines.append(f"    {field_name}: {pydantic_type} = default(True)")
            else:
                model_lines.append(f"    {field_name}: {pydantic_type}")

        # join the model lines and add to pydantic_models list
        pydantic_models.append("\n".join(model_lines))

    # return all models as a single string
    return "\n\n".join(pydantic_models)


# function to convert Prisma schema to SQLAlchemy models
def prisma_to_sqlalchemy(schema: str) -> str:
    models = model_regex.findall(schema)
    sqlalchemy_models = []

    for model_name, model_body in models:
        fields = field_regex.findall(model_body)
        model_lines = [f"class {model_name}(Base):"]
        model_lines.append(f'    __tablename__ = "{model_name}"\n')

        regular_fields = []
        foreign_key_fields = []
        relationship_fields = []

        for field_name, field_type, is_optional, field_attributes in fields:
            # handle one-to-many relationships (like messages Message[])
            if field_type.endswith("[]"):
                related_model = field_type[
                    :-2
                ]  # Get the related model name (remove [])
                relationship_field = (
                    f'    {field_name} = relationship("{related_model}",',
                    ' back_populates="{model_name.lower()}")',
                )
                relationship_fields.append(relationship_field)

                continue  # Skip processing this field as a regular column

            # convert Prisma type to SQLAlchemy type
            sqlalchemy_type = prisma_to_sqlalchemy_types.get(field_type, "String")

            # handle optional fields (marked with `?`)
            nullable = "nullable=True" if is_optional else "nullable=False"

            # handle primary keys and unique constraints
            primary_key = "@id" in field_attributes
            unique = "@unique" in field_attributes

            # handle default values (like cuid() and now())
            # COMMENT: we'll ignore default values for integers, since we do not insert values.
            # change it if you need to insert.
            # if '@default(cuid())' in field_attributes:
            #    default_value = 'default=str(uuid.uuid4())'
            if "@default(now())" in field_attributes:
                default_value = "default=datetime.now(timezone.utc)"
            elif "@default(false)" in field_attributes:
                default_value = 'default=False, server_default="0"'
            elif "@default(true)" in field_attributes:
                default_value = 'default=True, server_default="1"'
            else:
                default_value = None

            # handling ForeignKey relations
            if "@relation" in field_attributes:
                # extract foreign key relation from @relation directive
                relation_match = re.search(
                    r"fields: \[(\w+)\], references: \[(\w+)\]", field_attributes
                )
                if relation_match:
                    foreign_key_field = relation_match.group(1)
                    foreign_key_reference = relation_match.group(2)

                    column_params = f'mapped_column( ForeignKey("{field_type}.{foreign_key_reference}") )'
                    foreign_key_fields.append(
                        f"    {foreign_key_field} = {column_params}"
                    )

                    # add the relationship on the "many" side
                    relationship_field = (
                        f'    {field_type.lower()} = relationship("{field_type.capitalize()}"',
                        ', back_populates="{model_name.lower()}s")',
                    )
                    relationship_fields.append(relationship_field)

                    continue  # skip processing this field as a regular column

            column_params = f"Column({sqlalchemy_type}, "
            if primary_key:
                column_params += "primary_key=True, "
            if unique:
                column_params += "unique=True, "
            if nullable:
                column_params += f"{nullable}, "
            if default_value:
                column_params += f"{default_value}, "

            # final column definition
            column_params = column_params.rstrip(", ") + ")"
            regular_fields.append(f"    {field_name} = {column_params}")

        # add regular fields first and then foreign key fields
        model_lines.extend(regular_fields)
        model_lines.append("\n    # Foreign keys:")
        model_lines.extend(foreign_key_fields)
        model_lines.append("\n    # Relationships:")
        model_lines.extend(relationship_fields)

        # add the model lines to the final list of models
        sqlalchemy_models.append("\n".join(model_lines))

    # prepend with correct imports
    with open("workflows/resources/tables_imports.txt", "r") as file:
        imports = file.read()

    # return all models as a single string
    return imports + "\n\n" + "\n\n".join(sqlalchemy_models)


if __name__ == "__main__":
    # read prisma schema
    with open("workflows/resources/schema.prisma", "r") as schema:
        prisma_schema = schema.read()

    # generate Pydantic models from Prisma schema
    pydantic_output = prisma_to_pydantic(prisma_schema)
    sqlalchemy_output = prisma_to_sqlalchemy(prisma_schema)

    with open("./workflows/db/models.py", "w") as models:
        models.writelines(pydantic_output)

    with open("./workflows/db/tables.py", "w") as models:
        models.writelines(sqlalchemy_output)
