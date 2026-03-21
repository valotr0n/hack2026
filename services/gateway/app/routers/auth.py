from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth import create_access_token, hash_password, require_auth
from ..config import settings
from ..database import create_user, get_user_by_id, get_user_by_username

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    user_id: str
    username: str
    created_at: str


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Регистрация",
    description="Создаёт нового пользователя. Логин от 3 до 50 символов, пароль минимум 6 символов.",
)
async def register(req: RegisterRequest) -> UserResponse:
    if await get_user_by_username(settings.db_path, req.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Пользователь с таким именем уже существует",
        )
    user = await create_user(settings.db_path, req.username, hash_password(req.password))
    return UserResponse(user_id=user["id"], username=user["username"], created_at=user["created_at"])


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Вход",
    description="Возвращает JWT-токен. Передавайте его в заголовке `Authorization: Bearer <token>`. Токен действует 24 часа.",
)
async def login(req: LoginRequest) -> TokenResponse:
    user = await get_user_by_username(settings.db_path, req.username)
    if not user or user["password_hash"] != hash_password(req.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
        )
    return TokenResponse(access_token=create_access_token(user["id"]))


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Текущий пользователь",
    description="Возвращает данные авторизованного пользователя по токену.",
)
async def me(user_id: str = Depends(require_auth)) -> UserResponse:
    user = await get_user_by_id(settings.db_path, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден")
    return UserResponse(user_id=user["id"], username=user["username"], created_at=user["created_at"])
