from fastapi import APIRouter, status, Depends, HTTPException

from core import config

from ml import classifier, cropper
